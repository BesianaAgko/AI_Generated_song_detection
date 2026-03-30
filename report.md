# AI-Original Pairwise Similarity — Technical Report

## 1. Problem Statement

The goal is to design a system that ingests two full-length audio tracks and outputs a **Similarity (Attribution) Score** (0–1) reflecting the likelihood that Track B is an AI-generated derivative of Track A. This goes beyond binary classification — it requires **relational modeling** that remains robust to variations in tempo, arrangement, and spectral artifacts introduced by modern AI generators (Suno, Udio).

---

## 2. Solution Architecture

```
Track A (original)                Track B (suspected AI)
      │                                   │
      ▼                                   ▼
 Preprocessing                     Preprocessing
 librosa.load → 22050Hz mono       librosa.load → 22050Hz mono
      │                                   │
      ▼                                   ▼
  Chunking                           Chunking
 30s windows, 10s hop               30s windows, 10s hop
      │                                   │
      ▼                                   ▼
Feature Extraction                 Feature Extraction
  ├─ MFCCs (40 coefficients)         ├─ MFCCs (40)
  ├─ Chroma (12 pitch classes)       ├─ Chroma (12)
  ├─ Spectral Centroid               ├─ Spectral Centroid
  ├─ Spectral Flatness               ├─ Spectral Flatness
  ├─ Tempo (BPM)                     ├─ Tempo (BPM)
  ├─ HNR (Harmonic-to-Noise Ratio)   ├─ HNR
  ├─ Phase Discontinuity             ├─ Phase Discontinuity
  └─ CLAP embedding (512-dim)*       └─ CLAP embedding (512-dim)*
      │                                   │
      └───────────────┬───────────────────┘
                      ▼
           Similarity Engine
        Best-match per chunk strategy
        Weighted cosine / scalar similarity
                      │
                      ▼
           Attribution Score (0–1)
           + Per-feature breakdown
```

*CLAP is optional (requires ~1GB RAM, `--clap` flag).

---

## 3. Feature Engineering

### Why these features?

| Feature | Dimension | Rationale |
|---|---|---|
| **MFCCs** (mean + std) | 80 | Captures timbral texture — primary fingerprint of an instrument mix |
| **Chroma** (mean) | 12 | Pitch-class profile — melody/harmony similarity regardless of octave |
| **Tempo** (BPM) | 1 | Rhythmic matching; AI covers often preserve original tempo |
| **HNR** | 1 | Harmonic-to-Noise Ratio: AI generators produce characteristic HNR signatures |
| **Spectral Flatness** | 1 | Near-zero = tonal; near-one = noise-like; AI models show abnormal flatness patterns |
| **Phase Discontinuity** | 1 | AI vocoders/generators introduce unnatural phase jumps in the STFT |
| **CLAP embedding** | 512 | Semantic-level audio representation; strongest signal for high-level similarity |

### Are standard features sufficient?

Standard features (MFCCs, chroma) capture **timbre and harmony** but miss **AI-specific artifacts**. The inclusion of HNR, spectral flatness, and phase discontinuity addresses this gap. CLAP embeddings (when available) provide a neural, semantic-level comparison that correlates well with human perception.

### Feature weights

```
CLAP embedding:     0.40  (dominant — semantic similarity)
MFCCs:              0.20  (timbre fingerprint)
Chroma:             0.15  (melody/harmony)
HNR:                0.08  (AI artifact)
Spectral Flatness:  0.07  (AI artifact)
Phase Discontinuity:0.05  (AI artifact)
Tempo:              0.05  (rhythm)
```

When CLAP is unavailable, weights are renormalized across remaining features.

---

## 4. Handling Variable-Length Audio & Window-Bias Prevention

A naive approach would analyze only the first 30 seconds of each track. This system prevents window-bias through two mechanisms:

1. **Overlapping chunking**: Audio is segmented into 30s windows with a 10s hop, covering the entire track (a 3-minute track produces ~15 chunks).

2. **Best-match alignment strategy**: For each chunk in Track A, the system finds the *best-matching* chunk in Track B (cross-track maximum). The final score is the mean of all best matches. This handles:
   - Segment reordering (AI may rearrange sections)
   - Tempo shifts (matching chunks may be offset in time)
   - Intro/outro differences (unmatched sections don't penalize the score)

---

## 5. Similarity Computation

### Per-chunk similarity
- **Vector features** (MFCCs, chroma, CLAP): cosine similarity, normalized to [0, 1]
- **Scalar features** (tempo, HNR, spectral flatness, phase discontinuity): linear decay from 1.0 (identical) to 0.0 at a domain-specific `max_diff`

### Score interpretation
| Score | Interpretation |
|---|---|
| ≥ 0.85 | Very likely AI attribution — almost certainly an AI cover |
| 0.70–0.85 | Probable attribution — significant similarity |
| 0.50–0.70 | Ambiguous — moderate similarity |
| 0.30–0.50 | Low similarity — likely unrelated |
| < 0.30 | Unrelated tracks |

---

## 6. Datasets

| Dataset | Usage |
|---|---|
| **SONICS** (97k+ tracks) | Primary evaluation of AI-generated detection; covers Suno/Udio artifacts |
| **FakeMusicCaps** | Fine-grained artifact analysis on 10s clips; useful for calibrating HNR/phase thresholds |
| **MIPPIA SMP** | Ground-truth attribution pairs; ideal for tuning feature weights and score thresholds |

The MIPPIA dataset (Similar Music Pairs) is the most directly relevant: it provides known (original, similar) pairs that can be used to validate the attribution score against labeled data.

---

## 7. Tools & Libraries

| Category | Library | Purpose |
|---|---|---|
| Audio I/O & DSP | `librosa` | Loading, resampling, STFT, MFCCs, chroma, beat tracking |
| Audio I/O | `soundfile` | WAV/FLAC reading |
| Neural embeddings | `transformers` (CLAP) | Semantic audio embeddings via `laion/clap-htsat-unfused` |
| Deep learning | `torch` | CLAP inference backend |
| Similarity | `scikit-learn` | Cosine similarity |
| Numerics | `numpy`, `scipy` | Signal processing |

---

## 8. Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| 30s chunks with 10s hop | Covers full track, captures local patterns | Higher compute vs. global approach |
| Best-match (not DTW) | Simple, fast, handles reordering | May overestimate similarity for partially similar tracks |
| Cosine over Euclidean | Scale-invariant; better for high-dim embeddings | Loses magnitude information |
| CLAP optional | Allows use without GPU/large downloads | Without CLAP, detection is weaker for genre-level attribution |
| `max_diff` per scalar | Domain-adapted normalization | Requires domain knowledge to tune correctly |

---

## 9. Limitations

1. **No training data used**: The system is fully unsupervised — feature weights are hand-tuned. A supervised model trained on MIPPIA/SONICS pairs would significantly improve accuracy.

2. **CLAP dependency**: The strongest feature (CLAP embedding) requires ~1GB model download and GPU for reasonable speed at scale.

3. **No tempo normalization**: Chunks are compared at their original tempo. A track with significant BPM variation (e.g., a slowed/sped AI cover) may produce false negatives.

4. **Scalar feature sensitivity**: `max_diff` thresholds for HNR, spectral flatness, and phase discontinuity were set heuristically — they should be calibrated on labeled data.

5. **Stereo → mono collapse**: Stereo panning information is lost during preprocessing.

---

## 10. Future Improvements

1. **Supervised learning**: Train a binary/regression model on MIPPIA SMP with the extracted features as input — replace hand-tuned weights with learned ones.

2. **Dynamic Time Warping (DTW)**: Replace best-match with DTW for more principled alignment of time-shifted covers.

3. **MERT embeddings**: Use the music-specific MERT model for richer semantic embeddings than general-purpose CLAP.

4. **Tempo normalization**: Auto-detect BPM ratio between tracks and align chunks accordingly before comparison.

5. **Batch/dataset evaluation**: Wrap the pipeline in a dataset loader for bulk evaluation over SONICS/MIPPIA with precision/recall metrics.

6. **Explainability**: Add SHAP values or per-feature contribution visualization to explain *why* a score is high.

---

## 11. Usage

```bash
# Basic comparison (no CLAP)
python compare_tracks.py original.mp3 suspected_cover.mp3

# With CLAP neural embeddings (more accurate)
python compare_tracks.py original.mp3 suspected_cover.mp3 --clap

# JSON output
python compare_tracks.py original.mp3 suspected_cover.mp3 --clap --json
```

See `demo_notebook.ipynb` for end-to-end examples with synthetic audio signals.
