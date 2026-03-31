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

    heuristic_score = float(np.clip(
        sum(feature_scores[f] * w for f, w in _AI_DETECTION_WEIGHTS.items()),
        0.0, 1.0
    ))

    # ── Trained classifier (primary) ───────────────────────────────────────────
    if _TRAINED_DETECTOR is not None:
        pipeline     = _TRAINED_DETECTOR["pipeline"]
        feature_cols = _TRAINED_DETECTOR["feature_cols"]
        x = np.array([[track_means.get(col, 0.0) for col in feature_cols]])
        trained_ai_prob = float(pipeline.predict_proba(x)[0][1])
        ai_score = 0.75 * trained_ai_prob + 0.25 * heuristic_score

        # Guardrail: extremely AI-like spectral flatness should not be suppressed
        # by an underconfident classifier on clearly synthetic artifacts.
        if (
            feature_scores["spectral_flatness"] >= 0.60
            and heuristic_score >= 0.50
            and trained_ai_prob < 0.50
        ):
            ai_score = max(
                ai_score,
                0.60 + 0.25 * feature_scores["spectral_flatness"],
            )

        ai_score = round(float(np.clip(ai_score, 0.0, 1.0)), 4)

        if ai_score >= 0.75:
            interpretation = "Likely AI-generated"
        elif ai_score >= 0.55:
            interpretation = "Possibly AI-generated"
        elif ai_score >= 0.40:
            interpretation = "Ambiguous — borderline case"
        elif ai_score >= 0.20:
            interpretation = "Probably human-made"
        else:
            interpretation = "Likely human-made"

        return {
            "ai_score":      ai_score,
            "interpretation": interpretation,
            "method":        "hybrid_detector (LR + heuristic guardrail, CV F1=0.85)",
            "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
            "component_scores": {
                "trained_classifier": round(trained_ai_prob, 4),
                "heuristic": round(heuristic_score, 4),
            },
            "track_means":   {k: round(v, 6) for k, v in track_means.items()},
        }

    # ── Fallback: Gaussian heuristic ───────────────────────────────────────────
    ai_score = heuristic_score

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
    "clap":               0.35,  # Neural embedding — strongest signal when available
    "mfcc":               0.15,  # Timbre
    "mel":                0.10,  # Log-mel spectrogram summary
    "chroma":             0.15,  # Harmony / melody
    "tempo":              0.05,  # Rhythm
    "hnr":                0.08,  # Harmonic-to-Noise Ratio (AI artifact)
    "spectral_flatness":  0.07,  # Spectral flatness (AI artifact)
    "phase_discontinuity":0.05,  # Phase discontinuity (AI artifact)
}

_CONSISTENCY_WEIGHTS = {
    "mutual": 0.08,
    "coverage": 0.05,
    "order": 0.60,
    "margin": 0.05,
    "floor": 0.18,
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


def _best_chunk_matches(
    features_src: list[ChunkFeatures],
    features_dst: list[ChunkFeatures],
) -> tuple[list[float], list[int], list[float], dict[str, list[float]]]:
    """Match each source chunk to its best destination chunk."""
    match_scores: list[float] = []
    match_indices: list[int] = []
    match_margins: list[float] = []
    feature_totals: dict[str, list[float]] = {k: [] for k in WEIGHTS}

    for src_chunk in features_src:
        best_score = -1.0
        second_best_score = -1.0
        best_index = -1
        best_feat_scores: dict[str, float] = {}

        for dst_index, dst_chunk in enumerate(features_dst):
            feat_scores = _chunk_similarity(src_chunk, dst_chunk)
            total_score = _weighted_chunk_score(feat_scores)

            if total_score > best_score:
                second_best_score = best_score
                best_score = total_score
                best_index = dst_index
                best_feat_scores = feat_scores
            elif total_score > second_best_score:
                second_best_score = total_score

        match_scores.append(best_score)
        match_indices.append(best_index)
        match_margins.append(max(0.0, best_score - max(second_best_score, 0.0)))

        for feature_name, feature_score in best_feat_scores.items():
            if feature_score is not None:
                feature_totals[feature_name].append(feature_score)

    return match_scores, match_indices, match_margins, feature_totals


def _coverage_ratio(match_indices: list[int], target_len: int) -> float:
    """How much of the destination track is covered by distinct matches."""
    if target_len <= 0:
        return 0.0
    unique_matches = {index for index in match_indices if index >= 0}
    return float(np.clip(len(unique_matches) / target_len, 0.0, 1.0))


def _order_consistency(match_indices: list[int]) -> float:
    """Measures whether best-match chunk indices preserve global ordering."""
    if len(match_indices) < 2:
        return 0.0

    x = np.arange(len(match_indices), dtype=float)
    y = np.asarray(match_indices, dtype=float)
    if np.allclose(y, y[0]):
        return 0.0

    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))


def _mutual_match_ratio(matches_ab: list[int], matches_ba: list[int]) -> float:
    """Fraction of chunk matches that are mutual nearest neighbors."""
    if not matches_ab or not matches_ba:
        return 0.0

    mutual_matches = 0
    for index_a, index_b in enumerate(matches_ab):
        if 0 <= index_b < len(matches_ba) and matches_ba[index_b] == index_a:
            mutual_matches += 1

    normalizer = max(1, min(len(matches_ab), len(matches_ba)))
    return float(np.clip(mutual_matches / normalizer, 0.0, 1.0))


def _consistency_multiplier(
    mutual_ratio: float,
    coverage_ratio: float,
    order_consistency: float,
    margin_score: float,
) -> float:
    """Turns chunk-match diagnostics into a conservative global confidence factor."""
    weighted_sum = (
        _CONSISTENCY_WEIGHTS["mutual"] * mutual_ratio
        + _CONSISTENCY_WEIGHTS["coverage"] * coverage_ratio
        + _CONSISTENCY_WEIGHTS["order"] * (order_consistency ** 2)
        + _CONSISTENCY_WEIGHTS["margin"] * margin_score
        + _CONSISTENCY_WEIGHTS["floor"]
    )
    return float(np.clip(weighted_sum, 0.0, 1.0))


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
    if not features_a or not features_b:
        return {
            "attribution_score": 0.0,
            "interpretation": "Insufficient features for comparison",
            "chunk_scores": [],
            "feature_breakdown": {k: None for k in WEIGHTS},
            "diagnostics": {
                "base_similarity": 0.0,
                "mutual_match_ratio": 0.0,
                "coverage_ratio": 0.0,
                "order_consistency": 0.0,
                "margin_score": 0.0,
                "consistency_multiplier": 0.0,
            },
        }

    chunk_scores_ab, match_indices_ab, match_margins_ab, feature_totals_ab = _best_chunk_matches(
        features_a, features_b
    )
    chunk_scores_ba, match_indices_ba, match_margins_ba, feature_totals_ba = _best_chunk_matches(
        features_b, features_a
    )

    base_similarity = float(np.mean(chunk_scores_ab + chunk_scores_ba))
    mutual_ratio = _mutual_match_ratio(match_indices_ab, match_indices_ba)
    coverage_ratio = float(np.mean([
        _coverage_ratio(match_indices_ab, len(features_b)),
        _coverage_ratio(match_indices_ba, len(features_a)),
    ]))
    order_consistency = float(np.mean([
        _order_consistency(match_indices_ab),
        _order_consistency(match_indices_ba),
    ]))
    margin_score = float(np.clip(np.mean(match_margins_ab + match_margins_ba) / 0.20, 0.0, 1.0))
    consistency_multiplier = _consistency_multiplier(
        mutual_ratio,
        coverage_ratio,
        order_consistency,
        margin_score,
    )

    attribution_score = float(np.clip(base_similarity * consistency_multiplier, 0.0, 1.0))
    chunk_scores = chunk_scores_ab

    feature_breakdown = {}
    for feature_name in WEIGHTS:
        feature_values = feature_totals_ab[feature_name] + feature_totals_ba[feature_name]
        feature_breakdown[feature_name] = float(np.mean(feature_values)) if feature_values else None

    # Score interpretation
    if attribution_score >= 0.72:
        interpretation = "Strong evidence of relatedness — high pairwise similarity"
    elif attribution_score >= 0.55:
        interpretation = "Likely related tracks — significant similarity detected"
    elif attribution_score >= 0.40:
        interpretation = "Moderate similarity — ambiguous"
    elif attribution_score >= 0.30:
        interpretation = "Low similarity — likely unrelated tracks"
    else:
        interpretation = "Unrelated tracks"

    if verbose:
        print(f"\n{'='*50}")
        print(f"Attribution Score: {attribution_score:.3f}")
        print(f"Interpretation: {interpretation}")
        print(
            "Consistency diagnostics: "
            f"base={base_similarity:.3f}, mutual={mutual_ratio:.3f}, "
            f"coverage={coverage_ratio:.3f}, order={order_consistency:.3f}, "
            f"margin={margin_score:.3f}, factor={consistency_multiplier:.3f}"
        )
        print(f"\nPer-feature breakdown:")
        for k, v in feature_breakdown.items():
            bar = "#" * int((v or 0) * 20)
            print(f"  {k:8s}: {(v or 0):.3f}  {bar}")
        print(f"{'='*50}\n")

    return {
        "attribution_score": attribution_score,
        "interpretation":    interpretation,
        "chunk_scores":      chunk_scores,
        "feature_breakdown": feature_breakdown,
        "diagnostics": {
            "base_similarity": base_similarity,
            "mutual_match_ratio": mutual_ratio,
            "coverage_ratio": coverage_ratio,
            "order_consistency": order_consistency,
            "margin_score": margin_score,
            "consistency_multiplier": consistency_multiplier,
        },
    }