"""
feature_extractor.py
--------------------
Extracts audio features from a track in chunks.
Uses: librosa (MFCCs, mel-spectrogram, chroma, spectral), CLAP (neural embeddings)
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz - standard for music analysis
CHUNK_SEC     = 30      # window size in seconds
HOP_SEC       = 10      # overlap: each new chunk starts 10s after the previous one
N_MFCC        = 40      # number of MFCC coefficients
N_MELS        = 64      # mel bands for log-mel summary features
N_CHROMA      = 12      # 12 pitch classes (C, D, E, ...)


@dataclass
class ChunkFeatures:
    """All features for a 30s chunk."""
    chunk_id:         int
    start_sec:        float
    end_sec:          float

    # Timbre
    mfcc_mean:        np.ndarray = field(default_factory=lambda: np.array([]))  # (40,)
    mfcc_std:         np.ndarray = field(default_factory=lambda: np.array([]))  # (40,)
    mel_mean:         np.ndarray = field(default_factory=lambda: np.array([]))  # (64,)
    mel_std:          np.ndarray = field(default_factory=lambda: np.array([]))  # (64,)
    spectral_centroid:float = 0.0
    spectral_flatness:float = 0.0   # AI artifact detector

    # Melody & harmony
    chroma_mean:      np.ndarray = field(default_factory=lambda: np.array([]))  # (12,)
    tempo:            float = 0.0

    # AI artifact
    hnr:              float = 0.0   # Harmonic-to-Noise Ratio
    phase_discontinuity: float = 0.0

    # Neural embedding (CLAP) - optional, populated later
    clap_embedding:   Optional[np.ndarray] = None  # (512,)


class FeatureExtractor:
    """
    Loads an audio file, splits it into overlapping chunks,
    and extracts features from each chunk.
    """

    def __init__(self, use_clap: bool = False):
        """
        Args:
            use_clap: If True, loads the CLAP model (requires ~1GB RAM).
                      Set to False when testing for the first time.
        """
        self.use_clap = use_clap
        self._clap_model = None

        if use_clap:
            self._load_clap()

    # ── CLAP setup ────────────────────────────────────────────────────────────

    def _load_clap(self):
        """Loads the CLAP model from HuggingFace (only the first time)."""
        try:
            from transformers import ClapModel, ClapProcessor
            print("Loading CLAP model... (this may take a while)")
            self._clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
            self._clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
            self._clap_model.eval()
            print("CLAP ready!")
        except ImportError:
            print("WARNING: transformers was not found. Run: pip install transformers")
            self.use_clap = False

    def _get_clap_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Returns a 512-dim CLAP embedding for an audio chunk."""
        import torch
        inputs = self._clap_processor(
            audios=audio_chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )
        with torch.no_grad():
            emb = self._clap_model.get_audio_features(**inputs)
        return emb.squeeze().numpy()

    # ── Audio loading ─────────────────────────────────────────────────────────

    def load_audio(self, path: str) -> np.ndarray:
        """
        Loads an audio file and resamples it to SAMPLE_RATE.
        Supports mp3, wav, flac, ogg, etc.
        """
        print(f"Loading: {path}")
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / SAMPLE_RATE
        print(f"  Duration: {duration:.1f}s | Sample rate: {sr}Hz")
        return audio

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _make_chunks(self, audio: np.ndarray):
        """
                Splits the audio into overlapping windows.
                Returns a list of (start_sample, end_sample) tuples.

                Example for a 3-minute song:
          chunk 0: 0s–30s
          chunk 1: 10s–40s
          chunk 2: 20s–50s  ... κτλ
        """
        chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
        hop_samples   = int(HOP_SEC  * SAMPLE_RATE)
        total_samples = len(audio)

        chunks = []
        start = 0
        while start + chunk_samples <= total_samples:
            chunks.append((start, start + chunk_samples))
            start += hop_samples

        # Final chunk if the remaining audio is longer than 10s
        if start < total_samples and (total_samples - start) > 10 * SAMPLE_RATE:
            chunks.append((start, total_samples))

        print(f"  Created {len(chunks)} chunks ({CHUNK_SEC}s window, {HOP_SEC}s hop)")
        return chunks

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_hnr(self, audio_chunk: np.ndarray) -> float:
        """
        Harmonic-to-Noise Ratio: how "harmonic" the sound is.
        High HNR = clean harmonic sound (voice, violin).
        Low HNR = noise.
        AI generators tend to produce extreme values.
        """
        harmonic, percussive = librosa.effects.hpss(audio_chunk)
        harmonic_energy   = np.mean(harmonic ** 2) + 1e-10
        percussive_energy = np.mean(percussive ** 2) + 1e-10
        hnr = 10 * np.log10(harmonic_energy / percussive_energy)
        return float(hnr)

    def _compute_phase_discontinuity(self, audio_chunk: np.ndarray) -> float:
        """
        Measures how smoothly the phase changes in the spectrum.
        AI models often produce abrupt phase changes
        that do not appear in natural audio.
        """
        stft = librosa.stft(audio_chunk)
        phase = np.angle(stft)
        # Phase difference between consecutive frames
        phase_diff = np.diff(phase, axis=1)
        # Wrap to [-π, π]
        phase_diff = np.angle(np.exp(1j * phase_diff))
        discontinuity = float(np.mean(np.abs(phase_diff)))
        return discontinuity

    def _compute_chunk_features(
        self, chunk_audio: np.ndarray, chunk_id: int, start_s: float, end_s: float
    ) -> ChunkFeatures:
        """Computes all features for one chunk."""

        # MFCCs
        mfcc = librosa.feature.mfcc(y=chunk_audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)

        # Log-mel spectrogram summary
        mel_spec = librosa.feature.melspectrogram(
            y=chunk_audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Chroma (harmonic structure)
        chroma = librosa.feature.chroma_stft(y=chunk_audio, sr=SAMPLE_RATE)

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=chunk_audio, sr=SAMPLE_RATE)
        spec_flatness = librosa.feature.spectral_flatness(y=chunk_audio)

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=chunk_audio, sr=SAMPLE_RATE)

        features = ChunkFeatures(
            chunk_id          = chunk_id,
            start_sec         = start_s,
            end_sec           = end_s,
            mfcc_mean         = np.mean(mfcc, axis=1),
            mfcc_std          = np.std(mfcc, axis=1),
            mel_mean          = np.mean(log_mel, axis=1),
            mel_std           = np.std(log_mel, axis=1),
            spectral_centroid = float(np.mean(spec_centroid)),
            spectral_flatness = float(np.mean(spec_flatness)),
            chroma_mean       = np.mean(chroma, axis=1),
            tempo             = float(np.squeeze(tempo)),
            hnr               = self._compute_hnr(chunk_audio),
            phase_discontinuity = self._compute_phase_discontinuity(chunk_audio),
        )

        # CLAP embedding (if enabled)
        if self.use_clap and self._clap_model is not None:
            features.clap_embedding = self._get_clap_embedding(chunk_audio)

        return features

    # ── Main entry point ──────────────────────────────────────────────────────

    def extract(self, path: str) -> list[ChunkFeatures]:
        """
        Main function: loads a track and returns
        a list of ChunkFeatures for each chunk.

        Usage:
            extractor = FeatureExtractor(use_clap=True)
            features = extractor.extract("song.mp3")
        """
        audio  = self.load_audio(path)
        chunks = self._make_chunks(audio)

        all_features = []
        for i, (start, end) in enumerate(chunks):
            chunk_audio = audio[start:end]
            start_s     = start / SAMPLE_RATE
            end_s       = end   / SAMPLE_RATE

            feat = self._compute_chunk_features(chunk_audio, i, start_s, end_s)
            all_features.append(feat)

            if (i + 1) % 5 == 0:
                print(f"  Processing: {i+1}/{len(chunks)} chunks...")

        print(f"Done! Extracted {len(all_features)} chunks in total.")
        return all_features