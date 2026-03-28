# AI-Original Pairwise Similarity

A system for detecting AI-generated audio covers via pairwise similarity scoring.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic comparison (no neural embeddings)
python compare_tracks.py original.mp3 suspected_cover.mp3

# With CLAP neural embeddings (more accurate, requires ~1GB download)
python compare_tracks.py original.mp3 suspected_cover.mp3 --clap

# JSON output
python compare_tracks.py original.mp3 suspected_cover.mp3 --clap --json
```

## Example Output

```
==================================================
Attribution Score: 0.823
Interpretation: Probable attribution — significant similarity

Per-feature breakdown:
  clap              : 0.891  ██████████████████
  mfcc              : 0.812  ████████████████
  chroma            : 0.756  ███████████████
  tempo             : 0.950  ███████████████████
  hnr               : 0.634  ████████████
  spectral_flatness : 0.701  ██████████████
  phase_discontinuity: 0.580 ███████████
==================================================
```

## Score Interpretation

| Score | Interpretation |
|---|---|
| ≥ 0.85 | Very likely AI attribution — almost certainly an AI cover |
| 0.70–0.85 | Probable attribution — significant similarity |
| 0.50–0.70 | Ambiguous — moderate similarity |
| 0.30–0.50 | Low similarity — likely unrelated |
| < 0.30 | Unrelated tracks |

## Architecture

```
Track A (original)          Track B (suspected AI)
      │                              │
      ▼                              ▼
 Preprocessing               Preprocessing
 (22050Hz, mono)             (22050Hz, mono)
      │                              │
      ▼                              ▼
  Chunking                      Chunking
 (30s window, 10s hop)        (30s window, 10s hop)
      │                              │
      ▼                              ▼
Feature extraction           Feature extraction
  - MFCCs (40)                 - MFCCs (40)
  - Chroma (12)                - Chroma (12)
  - HNR, phase                 - HNR, phase
  - Spectral flatness          - Spectral flatness
  - CLAP (512)*                - CLAP (512)*
      │                              │
      └──────────┬───────────────────┘
                 ▼
        Similarity engine
        (best-match per chunk)
                 │
                 ▼
        Attribution Score (0–1)
```

*CLAP requires `--clap` flag.

## Datasets

- **SONICS**: `awsaf49/sonics` on HuggingFace (97k+ AI-generated tracks from Suno/Udio)
- **FakeMusicCaps**: Fine-grained text-to-music artifact analysis
- **MIPPIA SMP**: Attribution pairs for melodic similarity evaluation

## Files

| File | Description |
|------|-------------|
| `feature_extractor.py` | Audio loading, chunking, feature extraction |
| `similarity_engine.py` | Pairwise comparison, Attribution Score computation |
| `compare_tracks.py` | CLI entry point |
| `demo_notebook.ipynb` | End-to-end demo with synthetic audio signals |
| `report.md` | Full technical report |
