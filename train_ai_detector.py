"""
train_ai_detector.py
--------------------
Trains a Logistic Regression classifier to distinguish AI-generated
from human-made music tracks.

Training data:
  - Human: features/*.pkl (154 MIPPIA tracks)
  - AI:    ai_features_cache.pkl (101 FakeMusicCaps + SONICS tracks)

Saves the trained model to: ai_detector_model.pkl
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report

# ── Feature columns used for classification ────────────────────────────────────
FEATURE_COLS = [
    "spectral_flatness",
    "phase_discontinuity",
    "hnr_db",
    "zcr_mean",
    "spectral_centroid_hz",
    "tempo_bpm",
    "mfcc1_mean",
]

MODEL_FILE = Path("ai_detector_model.pkl")


def load_human_features() -> list[dict]:
    rows = []
    for pkl in Path("features").glob("*.pkl"):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict) and "hnr_db" in d:
            d["label"] = 0  # human
            rows.append(d)
    return rows


def load_ai_features() -> list[dict]:
    with open("ai_features_cache.pkl", "rb") as f:
        rows = pickle.load(f)
    for r in rows:
        r["label"] = 1  # AI
    return rows


def build_X_y(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r.get(col, 0.0) for col in FEATURE_COLS] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


def main():
    print("Loading features...")
    human = load_human_features()
    ai    = load_ai_features()
    print(f"  Human tracks: {len(human)}")
    print(f"  AI tracks:    {len(ai)}")

    all_rows = human + ai
    X, y = build_X_y(all_rows)
    print(f"  Total samples: {len(y)}  |  Features: {len(FEATURE_COLS)}")
    print(f"  Class balance: {int(y.sum())} AI / {int((y==0).sum())} human\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])

    # ── Cross-validation ───────────────────────────────────────────────────────
    print("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        return_train_score=False,
    )

    print(f"\n{'='*50}")
    print(f"Cross-validation results (5-fold):")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cv_results[f"test_{metric}"]
        print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"{'='*50}\n")

    # ── Train final model on all data ─────────────────────────────────────────
    pipeline.fit(X, y)

    print("Final model — training set classification report:")
    y_pred = pipeline.predict(X)
    print(classification_report(y, y_pred, target_names=["human", "AI"]))

    # ── Feature importance (log-reg coefficients) ──────────────────────────────
    coefs = pipeline.named_steps["clf"].coef_[0]
    print("Feature importance (coefficient magnitude):")
    for feat, coef in sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True):
        bar = "#" * int(abs(coef) * 5)
        print(f"  {feat:25s}: {coef:+.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_data = {
        "pipeline":     pipeline,
        "feature_cols": FEATURE_COLS,
        "n_human":      len(human),
        "n_ai":         len(ai),
        "cv_f1_mean":   cv_results["test_f1"].mean(),
        "cv_f1_std":    cv_results["test_f1"].std(),
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved: {MODEL_FILE}")
    print(f"CV F1: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")


if __name__ == "__main__":
    main()
