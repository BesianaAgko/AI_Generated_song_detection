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


def _load_trained_detector():
    """Load the trained Logistic Regression model if available."""
    import pickle
    from pathlib import Path
    model_path = Path(__file__).parent / "ai_detector_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


_TRAINED_DETECTOR = _load_trained_detector()


def ai_detection_score(features: list[ChunkFeatures]) -> dict:
    """
    Estimates the likelihood that a track is AI-generated.

    Uses a trained Logistic Regression classifier (F1=0.85 on 5-fold CV)
    trained on 154 human (MIPPIA) and 101 AI (FakeMusicCaps + SONICS) tracks.

    Falls back to Gaussian likelihood ratio heuristic if model is unavailable.

    Returns:
        dict with:
          - ai_score: float 0–1  (1.0 = AI-generated)
          - interpretation: str
          - method: 'trained_classifier' or 'heuristic'
          - feature_scores: per-feature AI likelihood (heuristic only)
    """
    if not features:
        return {"ai_score": 0.0, "interpretation": "No features", "method": "none"}

    # Aggregate features across all chunks
    sf_vals  = [f.spectral_flatness   for f in features]
    pd_vals  = [f.phase_discontinuity for f in features]
    hnr_vals = [f.hnr                 for f in features]
    sc_vals  = [f.spectral_centroid   for f in features]
    tp_vals  = [f.tempo               for f in features]

    track_means = {
        "spectral_flatness":   float(np.mean(sf_vals)),
        "phase_discontinuity": float(np.mean(pd_vals)),
        "hnr_db":              float(np.mean(hnr_vals)),
        "spectral_centroid_hz":float(np.mean(sc_vals)),
        "tempo_bpm":           float(np.mean(tp_vals)),
        "zcr_mean":            0.0,   # not available per-chunk, use 0
        "mfcc1_mean":          float(np.mean([f.mfcc_mean[0] for f in features if f.mfcc_mean.size > 0])),
    }

    # ── Trained classifier (primary) ───────────────────────────────────────────
    if _TRAINED_DETECTOR is not None:
        pipeline     = _TRAINED_DETECTOR["pipeline"]
        feature_cols = _TRAINED_DETECTOR["feature_cols"]
        x = np.array([[track_means.get(col, 0.0) for col in feature_cols]])
        ai_prob  = float(pipeline.predict_proba(x)[0][1])
        ai_score = round(ai_prob, 4)

        if ai_score >= 0.80:
            interpretation = "Likely AI-generated"
        elif ai_score >= 0.60:
            interpretation = "Possibly AI-generated"
        elif ai_score >= 0.40:
            interpretation = "Ambiguous — borderline case"
        elif ai_score >= 0.20:
            interpretation = "Likely human-made"
        else:
            interpretation = "Likely human-made"

        return {
            "ai_score":      ai_score,
            "interpretation": interpretation,
            "method":        "trained_classifier (LR, CV F1=0.85)",
            "track_means":   {k: round(v, 6) for k, v in track_means.items()},
        }

    # ── Fallback: Gaussian heuristic ───────────────────────────────────────────
    heuristic_vals = {
        "spectral_flatness":   track_means["spectral_flatness"],
        "phase_discontinuity": track_means["phase_discontinuity"],
        "hnr":                 track_means["hnr_db"],
    }
    feature_scores = {}
    for feat, val in heuristic_vals.items():
        p_ai    = _gaussian_pdf(val, _AI_PROFILE[feat]["mean"],    _AI_PROFILE[feat]["std"])
        p_human = _gaussian_pdf(val, _HUMAN_PROFILE[feat]["mean"], _HUMAN_PROFILE[feat]["std"])
        feature_scores[feat] = float(p_ai / (p_ai + p_human + 1e-10))

    ai_score = float(np.clip(
        sum(feature_scores[f] * w for f, w in _AI_DETECTION_WEIGHTS.items()),
        0.0, 1.0
    ))

    if ai_score >= 0.80:
        interpretation = "Likely AI-generated"
    elif ai_score >= 0.60:
        interpretation = "Probably AI-generated"
    elif ai_score >= 0.40:
        interpretation = "Ambiguous — could be AI or human"
    elif ai_score >= 0.20:
        interpretation = "Probably human-made"
    else:
        interpretation = "Likely human-made"

    return {
        "ai_score":      round(ai_score, 4),
        "interpretation": interpretation,
        "method":        "heuristic (Gaussian profile)",
        "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
        "track_means":   {k: round(v, 6) for k, v in track_means.items()},
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

    # Phase discontinuity (AI artifact — typical track-to-track diff is ~0.05)
    scores["phase_discontinuity"] = _scalar_sim(
        fa.phase_discontinuity, fb.phase_discontinuity, max_diff=0.05
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
        interpretation = "Strong evidence of relatedness — high pairwise similarity"
    elif attribution_score >= 0.70:
        interpretation = "Likely related tracks — significant similarity detected"
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