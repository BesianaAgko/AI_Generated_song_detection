"""
similarity_engine.py
--------------------
Compares the features of two tracks and produces an Attribution Score (0.0 – 1.0).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import ChunkFeatures


# ── AI vs Human reference profiles ───────────────────────────────────────────
# Derived empirically: SONICS (30 AI tracks) vs MIPPIA features/ (154 human tracks)
_AI_PROFILE = {
    "spectral_flatness":   {"mean": 0.000939, "std": 0.001659},
    "phase_discontinuity": {"mean": 1.5683,   "std": 0.0046},
    "hnr":                 {"mean": 5.55,      "std": 3.04},
}
_HUMAN_PROFILE = {
    "spectral_flatness":   {"mean": 0.025170,  "std": 0.019234},
    "phase_discontinuity": {"mean": 1.5628,    "std": 0.014614},
    "hnr":                 {"mean": 4.93,       "std": 2.734},
}
# Weights per feature for AI detection (spectral_flatness is strongest discriminator)
_AI_DETECTION_WEIGHTS = {
    "spectral_flatness":   0.60,
    "phase_discontinuity": 0.30,
    "hnr":                 0.10,
}


def _gaussian_pdf(x: float, mean: float, std: float) -> float:
    """Gaussian probability density (unnormalized)."""
    return float(np.exp(-0.5 * ((x - mean) / (std + 1e-10)) ** 2))


def ai_detection_score(features: list[ChunkFeatures]) -> dict:
    """
    Estimates the likelihood that a track is AI-generated.

    Compares per-chunk features against empirical AI (SONICS) and
    human (MIPPIA) reference profiles using Gaussian likelihood ratios.

    Returns:
        dict with:
          - ai_score: float 0–1  (1.0 = almost certainly AI-generated)
          - interpretation: str
          - feature_scores: per-feature AI likelihood
    """
    if not features:
        return {"ai_score": 0.0, "interpretation": "No features", "feature_scores": {}}

    # Aggregate scalar features across all chunks
    sf_vals   = [f.spectral_flatness   for f in features]
    pd_vals   = [f.phase_discontinuity for f in features]
    hnr_vals  = [f.hnr                 for f in features]

    track_vals = {
        "spectral_flatness":   float(np.mean(sf_vals)),
        "phase_discontinuity": float(np.mean(pd_vals)),
        "hnr":                 float(np.mean(hnr_vals)),
    }

    # Per-feature Gaussian likelihood ratio: P(AI) / (P(AI) + P(human))
    feature_scores = {}
    for feat, val in track_vals.items():
        p_ai    = _gaussian_pdf(val, _AI_PROFILE[feat]["mean"],    _AI_PROFILE[feat]["std"])
        p_human = _gaussian_pdf(val, _HUMAN_PROFILE[feat]["mean"], _HUMAN_PROFILE[feat]["std"])
        feature_scores[feat] = float(p_ai / (p_ai + p_human + 1e-10))

    # Weighted combination
    ai_score = float(np.clip(
        sum(feature_scores[f] * w for f, w in _AI_DETECTION_WEIGHTS.items()),
        0.0, 1.0
    ))

    if ai_score >= 0.80:
        interpretation = "Very likely AI-generated"
    elif ai_score >= 0.60:
        interpretation = "Probably AI-generated"
    elif ai_score >= 0.40:
        interpretation = "Ambiguous — could be AI or human"
    elif ai_score >= 0.20:
        interpretation = "Probably human-made"
    else:
        interpretation = "Very likely human-made"

    return {
        "ai_score":      round(ai_score, 4),
        "interpretation": interpretation,
        "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
        "track_means":   {k: round(v, 6) for k, v in track_vals.items()},
    }


# ── Weights for each feature group ───────────────────────────────────────────
# Sum = 1.0
WEIGHTS = {
    "clap":               0.35,  # Neural embedding — πιο σημαντικό
    "mfcc":               0.15,  # Timbre
    "mel":                0.10,  # Log-mel spectrogram summary
    "chroma":             0.15,  # Harmony / melody
    "tempo":              0.05,  # Rhythm
    "hnr":                0.08,  # Harmonic-to-Noise Ratio (AI artifact)
    "spectral_flatness":  0.07,  # Spectral flatness (AI artifact)
    "phase_discontinuity":0.05,  # Phase discontinuity (AI artifact)
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0–1."""
    if a.size == 0 or b.size == 0:
        return 0.0
    sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    # cosine is -1..1, convert it to 0..1
    return float((sim[0, 0] + 1.0) / 2.0)


def _scalar_sim(a: float, b: float, max_diff: float = 100.0) -> float:
    """Similarity for scalar values (e.g. tempo, HNR). Returns 0–1."""
    diff = abs(a - b)
    return float(max(0.0, 1.0 - diff / max_diff))


def _chunk_similarity(fa: ChunkFeatures, fb: ChunkFeatures) -> dict[str, float]:
    """
    Computes per-feature-group similarity between two chunks.
    Returns a dict with one score per group.
    """
    scores = {}

    # CLAP embedding (if available)
    if fa.clap_embedding is not None and fb.clap_embedding is not None:
        scores["clap"] = _cosine(fa.clap_embedding, fb.clap_embedding)
    else:
        # Fallback: redistribute the weight across the remaining features
        scores["clap"] = None

    # MFCCs (combine mean and std)
    mfcc_a = np.concatenate([fa.mfcc_mean, fa.mfcc_std])
    mfcc_b = np.concatenate([fb.mfcc_mean, fb.mfcc_std])
    scores["mfcc"] = _cosine(mfcc_a, mfcc_b)

    # Log-mel spectrogram (combine mean and std)
    mel_a = np.concatenate([fa.mel_mean, fa.mel_std])
    mel_b = np.concatenate([fb.mel_mean, fb.mel_std])
    scores["mel"] = _cosine(mel_a, mel_b)

    # Chroma
    scores["chroma"] = _cosine(fa.chroma_mean, fb.chroma_mean)

    # Tempo (max_diff = 60 BPM for a reasonable range)
    scores["tempo"] = _scalar_sim(fa.tempo, fb.tempo, max_diff=60.0)

    # HNR (AI artifact — max_diff = 20 dB)
    scores["hnr"] = _scalar_sim(fa.hnr, fb.hnr, max_diff=20.0)

    # Spectral flatness (AI artifact — range 0–1)
    scores["spectral_flatness"] = _scalar_sim(
        fa.spectral_flatness, fb.spectral_flatness, max_diff=1.0
    )

    # Phase discontinuity (AI artifact — range 0–π)
    scores["phase_discontinuity"] = _scalar_sim(
        fa.phase_discontinuity, fb.phase_discontinuity, max_diff=3.14159
    )

    return scores


def _weighted_chunk_score(scores: dict[str, float]) -> float:
    """
    Combines per-feature scores using WEIGHTS.
    If CLAP is unavailable, redistributes its weight.
    """
    if scores["clap"] is None:
        # Redistribution: normalize over the weights of active features
        active_features = {k: v for k, v in scores.items() if v is not None}
        total_weight = sum(WEIGHTS[k] for k in active_features) or 1.0
        total_score = sum(
            scores[k] * (WEIGHTS[k] / total_weight)
            for k in active_features
        )
    else:
        total_score = sum(
            scores[k] * WEIGHTS[k]
            for k in WEIGHTS
            if scores.get(k) is not None
        )

    return float(np.clip(total_score, 0.0, 1.0))


def compare_tracks(
    features_a: list[ChunkFeatures],
    features_b: list[ChunkFeatures],
    verbose: bool = False
) -> dict:
    """
        Main comparison function.

        Strategy:
            - Each chunk from A is compared against each chunk from B.
            - For each chunk in A, keep the best match from B.
            - The final score is the mean of all best matches.
            This handles tempo shifts, arrangement changes, and segment reordering.

    Returns:
                dict with:
          - attribution_score: float 0–1
          - interpretation: str
                    - chunk_scores: list of floats (one per chunk of A)
                    - feature_breakdown: mean score per feature group
    """
    chunk_scores     = []
    feature_totals: dict[str, list] = {k: [] for k in WEIGHTS}

    for fa in features_a:
        best_score        = -1.0
        best_feat_scores  = {}

        for fb in features_b:
            feat_scores  = _chunk_similarity(fa, fb)
            total        = _weighted_chunk_score(feat_scores)

            if total > best_score:
                best_score       = total
                best_feat_scores = feat_scores

        chunk_scores.append(best_score)
        for k, v in best_feat_scores.items():
            if v is not None:
                feature_totals[k].append(v)

    attribution_score = float(np.mean(chunk_scores))

    # Score interpretation
    if attribution_score >= 0.85:
        interpretation = "Very likely attribution — almost certainly an AI cover"
    elif attribution_score >= 0.70:
        interpretation = "Probable attribution — significant similarity"
    elif attribution_score >= 0.50:
        interpretation = "Moderate similarity — ambiguous"
    elif attribution_score >= 0.30:
        interpretation = "Low similarity — likely unrelated tracks"
    else:
        interpretation = "Unrelated tracks"

    feature_breakdown = {
        k: float(np.mean(v)) if v else None
        for k, v in feature_totals.items()
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Attribution Score: {attribution_score:.3f}")
        print(f"Interpretation: {interpretation}")
        print(f"\nPer-feature breakdown:")
        for k, v in feature_breakdown.items():
            bar = "█" * int((v or 0) * 20)
            print(f"  {k:8s}: {(v or 0):.3f}  {bar}")
        print(f"{'='*50}\n")

    return {
        "attribution_score": attribution_score,
        "interpretation":    interpretation,
        "chunk_scores":      chunk_scores,
        "feature_breakdown": feature_breakdown,
    }