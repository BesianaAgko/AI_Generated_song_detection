"""
similarity_engine.py
--------------------
Compares the features of two tracks and produces an Attribution Score (0.0 – 1.0).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import ChunkFeatures


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