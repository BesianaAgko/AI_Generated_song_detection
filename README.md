# AI-Generated Song Detection & Attribution

A Python system that takes two full-length audio files and outputs:
1. **Attribution Score** — likelihood that Track B is derived from Track A
2. **AI Detection Score** — likelihood that a track is AI-generated (Suno, Udio, etc.)

The current version uses:
- a **consistency-aware attribution scorer** that penalizes accidental local matches
- a **hybrid AI detector** that blends a trained Logistic Regression model with feature-based artifact guardrails

---

## Quick Start

```bash
# Compare two tracks
python compare_tracks.py original.mp3 suspected_cover.mp3

# With CLAP neural embeddings (more accurate, requires ~1GB download)
python compare_tracks.py original.mp3 suspected_cover.mp3 --clap

# JSON output
python compare_tracks.py original.mp3 suspected_cover.mp3 --json
```

**Example output:**
```
Track A: original.mp3
Track B: suspected_cover.mp3

==================================================
Attribution Score: 0.707 — Likely related tracks — significant similarity detected
Consistency diagnostics: base=0.958, order=0.950, coverage=0.219, factor=0.739

Per-feature breakdown:
  mfcc    : 0.979  ████████████████████
  mel     : 0.997  ████████████████████
  chroma  : 0.984  ████████████████████
  tempo   : 0.784  ████████████████
  hnr     : 0.889  █████████████████
  ...
==================================================

AI Detection — Track A: 0.086 → Likely human-made  [hybrid_detector]
AI Detection — Track B: 0.768 → Likely AI-generated [hybrid_detector]
==================================================
```

---

## System Architecture

```
Track A                           Track B
   │                                 │
   ▼                                 ▼
librosa.load → 22050Hz mono     librosa.load → 22050Hz mono
   │                                 │
   ▼                                 ▼
Chunking: 30s windows, 10s hop  Chunking: 30s windows, 10s hop
   │                                 │
   ▼                                 ▼
Feature Extraction               Feature Extraction
  ├─ MFCCs (40 coefficients)       ├─ MFCCs (40)
  ├─ Log-Mel Spectrogram (64)      ├─ Log-Mel (64)
  ├─ Chroma (12 pitch classes)     ├─ Chroma (12)
  ├─ Tempo (BPM)                   ├─ Tempo
  ├─ HNR (Harmonic-to-Noise)       ├─ HNR
  ├─ Spectral Flatness             ├─ Spectral Flatness
  ├─ Phase Discontinuity           ├─ Phase Discontinuity
  └─ CLAP embedding (512-dim)*     └─ CLAP embedding*
   │                                 │
   └──────────────┬──────────────────┘
                  ▼
        Similarity Engine
  Bidirectional chunk matching +
  global consistency diagnostics
  (order / coverage / mutuality)
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
  Attribution Score     AI Detection Score
     (0.0 – 1.0)      Hybrid detector
                    Logistic Regression +
                    heuristic artifact guardrail
                      trained on 255 tracks
                        (CV F1 = 0.85)
```

*CLAP is optional — use `--clap` flag.

---

## Window-Bias Prevention

A naive approach analyzes only the first 30 seconds. This system avoids that through:

1. **Overlapping chunks**: 30s windows with 10s hop → a 3-minute track produces ~15 chunks covering the entire audio
2. **Bidirectional matching**: chunks from A are matched against B and chunks from B are matched against A
3. **Consistency-aware scoring**: the final attribution score is reduced when the best matches are not globally ordered or do not cover the destination track consistently

---

## AI Detection (Task 1)

The `ai_detection_score()` function uses a **hybrid detector**:
- a trained **Logistic Regression classifier** over track-level summary features
- a **feature-based heuristic guardrail** using spectral flatness, phase discontinuity, and HNR
- a safety override for strongly synthetic artifact patterns that the classifier underestimates

| | Value |
|---|---|
| Training samples | 255 (154 human + 101 AI) |
| Human source | MIPPIA SMP dataset |
| AI source | FakeMusicCaps + SONICS (Suno/Udio) |
| Features | spectral_flatness, zcr, mfcc1, spectral_centroid, hnr, tempo, phase_discontinuity |
| Cross-validation | 5-fold stratified |
| **CV F1** | **0.85 ± 0.06** |
| **CV Recall** | **0.96 ± 0.08** |

Current demo thresholds:
- `>= 0.75` → likely AI-generated
- `>= 0.55` → possibly AI-generated
- `< 0.20` → likely human-made

Sanity-check examples from the current build:
- Jennifer Rush original: `0.0862`
- Céline Dion remake: `0.0532`
- SONICS Suno track: `0.7679`

---

## Attribution (Task 2 — Bonus)

The attribution scorer was recently recalibrated to reduce false positives on unrelated tracks by adding a global consistency factor on top of local chunk similarity.

Current demo threshold:
- `>= 0.55` → related tracks

Sanity-check examples from the current build:
- Positive pair: Jennifer Rush original ↔ Céline Dion remake = `0.7074`
- Negative pair: Jennifer Rush original ↔ SONICS Suno track = `0.3168`
- Negative pair: Jennifer Rush original ↔ Summer Dream = `0.2494`

Important: the existing [mippia_results.json](c:\Users\up105\Documents\Orfium_challenge\AI_Generated_song_detection\mippia_results.json) and [evaluation_metrics.json](c:\Users\up105\Documents\Orfium_challenge\AI_Generated_song_detection\evaluation_metrics.json) were generated before this scorer recalibration. Re-run the evaluation pipeline below if you want fully up-to-date benchmark numbers.

---

## Project Structure

```
.
├── Assesment/
│   ├── IN-Hiring Assesment - AI Generated song detection.pdf  # Original assessment brief
│   └── report.pdf                                             # Technical report (PDF)
├── compare_tracks.py        # CLI entry point
├── feature_extractor.py     # Audio loading, chunking, feature extraction
├── similarity_engine.py     # Attribution score + AI detection score
├── train_ai_detector.py     # Train Logistic Regression AI detector
├── evaluate_mippia.py       # Run attribution on full MIPPIA dataset
├── evaluate_attribution.py  # Precision/Recall/F1 evaluation
├── ai_detector_model.pkl    # Trained model (generated by train_ai_detector.py)
├── mippia_results.json      # Attribution results on MIPPIA pairs
├── evaluation_metrics.json  # Precision/Recall/F1 at multiple thresholds
├── data_exploration.ipynb   # Dataset analysis (Parts 1–5)
├── demo_notebook.ipynb      # End-to-end demo with real audio
├── report.md                # Technical report (source)
└── requirements.txt         # Dependencies
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AI_Generated_song_detection.git
cd AI_Generated_song_detection

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Full Pipeline

> **Prerequisites:** Download the datasets first — see the [Datasets](#datasets) section below for instructions on where to place each one.

```bash
# Step 1 — Evaluate attribution on all complete MIPPIA pairs
#    This also generates the features/ directory (human feature cache)
#    needed by the AI detector training in the next step.
#    Results are saved incrementally to mippia_results.json
#    (safe to interrupt and resume)
python evaluate_mippia.py

# Step 2 — Train the AI detector (run once, after Step 1)
#    Requires: features/ (from Step 1) and data/FakeMusicCaps/ + data/sonics/
#    Generates: ai_detector_model.pkl and ai_features_cache.pkl
python train_ai_detector.py

# Step 3 — Compute precision/recall/F1 from the attribution results
python evaluate_attribution.py

# If you modify similarity_engine.py, rerun Steps 1 and 3 to refresh
# mippia_results.json and evaluation_metrics.json with the new scorer.

# Step 4 — Explore the datasets (optional but recommended)
#    Open and run all cells in:
#    - data_exploration.ipynb  (dataset analysis, Parts 1–5)
#    - demo_notebook.ipynb     (end-to-end demo with real audio)

# Step 5 — Compare any two tracks
python compare_tracks.py "path/to/original.wav" "path/to/cover.wav"

# Example with a real MIPPIA pair:
python compare_tracks.py \
  "smp_dataset/final_dataset/5/Jennifer Rush - The Power Of Love _Official Video_ _VOD_.wav" \
  "smp_dataset/final_dataset/5/Céline Dion - The Power Of Love _Official Remastered HD Video_.wav"
```

---

## Just want to compare two tracks (no datasets)?

If you only have your own audio files and want to skip the full pipeline:

```bash
# train_ai_detector.py requires the features/ and data/ directories.
# Without them, compare_tracks.py will still work but AI detection
# will fall back to the heuristic profile instead of the hybrid detector.

python compare_tracks.py "path/to/original.mp3" "path/to/cover.mp3"
```

Supported formats: `.mp3` `.wav` `.flac` `.ogg` `.m4a`
Minimum recommended duration: **~40 seconds** per track.

---

### Expected directory structure before running

```
AI_Generated_song_detection/
├── smp_dataset/
│   ├── Final_dataset_pairs.csv
│   └── final_dataset/        ← downloaded by smp_dataset/download.py
├── data/
│   ├── sonics/
│   │   └── fake_songs/       ← downloaded from HuggingFace
│   └── FakeMusicCaps/
│       └── FakeMusicCaps.zip ← downloaded manually
└── features/                 ← auto-generated by evaluate_mippia.py
```

---

## Datasets

All three datasets are excluded from the repository (`.gitignore`). Below are the instructions to reproduce the local setup.

| Dataset | Role | Size |
|---|---|---|
| [MIPPIA SMP](https://github.com/Mippia/smp_dataset) | Attribution evaluation — labeled (original, similar) pairs | 158 pairs |
| [SONICS](https://huggingface.co/datasets/awsaf49/sonics) | AI-generated tracks (Suno/Udio) for detector training | 97k+ tracks |
| [FakeMusicCaps](https://github.com/LucasPLopes/FakeMusicCaps) | Text-to-music clips from 5 models for artifact analysis | 55k clips |

### MIPPIA SMP

Audio files are downloaded from YouTube via the dataset's own `download.py` script using `yt-dlp`.

```bash
# 1. Clone the dataset repo
git clone https://github.com/Mippia/smp_dataset.git

# 2. Install dependencies
pip install yt-dlp pandas ffmpeg-python

# 3. Run the download script (downloads all pairs as WAV)
cd smp_dataset
python download.py
```

Expected structure after download:
```
smp_dataset/
├── Final_dataset_pairs.csv
└── final_dataset/
    ├── 1/
    │   ├── Summer Dream.wav
    │   └── 김민종...wav
    ├── 2/
    │   └── ...
    └── ...
```

> Of 158 pairs, ~62 were fully downloadable at time of evaluation (YouTube availability varies).

---

### SONICS

Hosted on HuggingFace. Can be used in streaming mode (no download) or downloaded locally.

```python
# Streaming (metadata only — no disk space needed)
from datasets import load_dataset
sonics = load_dataset("awsaf49/sonics", split="train", streaming=True)

# Full local download (~several GB)
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="awsaf49/sonics",
    repo_type="dataset",
    local_dir="data/sonics"
)
```

Audio files are stored as zip archives (`part_01.zip` … `part_N.zip`) under `data/sonics/fake_songs/`. The system samples from these zips directly without full extraction.

---

### FakeMusicCaps

Download the zip from the [FakeMusicCaps repository](https://github.com/LucasPLopes/FakeMusicCaps) and place it at `data/FakeMusicCaps/FakeMusicCaps.zip`.

```
data/
└── FakeMusicCaps/
    └── FakeMusicCaps.zip   ← place here
```

The system extracts a sample (~20 files per model) automatically on first use. To extract manually:

```python
import zipfile
from pathlib import Path

with zipfile.ZipFile("data/FakeMusicCaps/FakeMusicCaps.zip") as z:
    z.extractall("data/FakeMusicCaps/")
```

Expected structure after extraction:
```
data/FakeMusicCaps/
├── audioldm2/
├── MusicGen_medium/
├── musicldm/
├── mustango/
└── stable_audio_open/
```

---

## Feature Engineering

| Feature | Dim | Rationale |
|---|---|---|
| MFCCs (mean + std) | 80 | Timbral texture — primary audio fingerprint |
| Log-Mel Spectrogram | 128 | Frequency content summary |
| Chroma | 12 | Pitch-class profile — melody/harmony |
| Tempo | 1 | Rhythm; AI covers often preserve original BPM |
| HNR | 1 | AI generators produce characteristic harmonic signatures |
| Spectral Flatness | 1 | Near-zero in AI output; higher variance in human recordings |
| Phase Discontinuity | 1 | AI vocoders introduce unnatural STFT phase jumps |
| CLAP embedding* | 512 | Semantic-level similarity (strongest signal) |

---

## Limitations

- **Training set size**: AI detector trained on 255 samples — larger dataset would improve generalization
- **No tempo normalization**: Significantly sped-up/slowed-down covers may score lower
- **CLAP optional**: Without neural embeddings, purely spectral features are used
- **Attribution ≠ AI detection**: High attribution score means the tracks are related, not necessarily that one is AI-generated
- **Thresholds are currently demo-calibrated**: after the scorer update, a full benchmark rerun is still recommended for final reporting

---

## Design Decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| Bidirectional chunk matching + consistency factor | Keeps local flexibility but penalizes accidental matches | Adds more tuning and diagnostics |
| Hybrid AI detector | Keeps LR probabilities but protects against missed synthetic artifacts | More heuristic logic to maintain |
| Cosine similarity | Scale-invariant for high-dim embeddings | Loses magnitude info |
| CLAP optional | Works without GPU | Weaker without neural embeddings |
