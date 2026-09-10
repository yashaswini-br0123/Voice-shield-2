import numpy as np
from typing import Dict, Any, Tuple
from backend.audio.feature_extraction import extract_acoustic_features
from backend.config import settings


class AcousticDetector:
    """
    Layer 1: Acoustic Analysis Module
    Analyzes high-level acoustic features: MFCCs, Mel Spectrogram, Pitch/F0 trajectory,
    spectral centroid, bandwidth, zero-crossing rate, energy variance, and room acoustic proxies.
    """
    def __init__(self):
        self.gemini_api_key = settings.GEMINI_API_KEY

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Tuple[float, Dict[str, Any]]:
        """
        Runs acoustic feature extraction and computes synthetic voice probability [0.0 - 1.0].
        Returns probability score and technical acoustic findings breakdown.
        """
        features = extract_acoustic_features(audio, sr)
        
        # Calculate anomaly scores based on acoustic inconsistency heuristics
        anomaly_reasons = []
        scores = []
        
        # 1. Pitch Jitter & Stability
        # Human speech exhibits micro-vibrato (jitter ~0.005-0.03). Synthetic voices often show unnaturally static F0 or step jumps.
        f0_jitter = features.get("f0_jitter", 0.0)
        pitch_std = features.get("pitch_std", 0.0)
        voiced_ratio = features.get("voiced_ratio", 0.0)
        
        if voiced_ratio > 0.2:
            if f0_jitter < 0.0015:
                scores.append(0.85)
                anomaly_reasons.append("Unnaturally rigid pitch contour with near-zero micro-vibrato (jitter < 0.15%)")
            elif f0_jitter > 0.15:
                scores.append(0.78)
                anomaly_reasons.append("Unstable glottal pitch transitions typical of voice conversion artifacts")
            else:
                scores.append(0.18)
        else:
            scores.append(0.20)

        # 2. High-Frequency Spectral Cutoff
        # Many neural vocoders hard-cutoff energy near vocoder frequency in high sample rates
        high_freq_ratio = features.get("high_freq_energy_ratio", 0.0)
        if sr >= 22050 and high_freq_ratio < 1e-6:
            scores.append(0.78)
            anomaly_reasons.append("Abrupt high-frequency attenuation above 7.5 kHz (neural vocoder filter signature)")
        else:
            scores.append(0.20)

        # 3. Spectral Enveloping & Bandwidth Oversmoothing
        mel_std = features.get("mel_std", 10.0)
        bandwidth_mean = features.get("bandwidth_mean", 1000.0)
        if mel_std < 3.0:
            scores.append(0.75)
            anomaly_reasons.append("Oversmoothed Mel-spectrogram energy distribution across frequency bands")
        else:
            scores.append(0.20)

        # 4. Spectral Flatness
        spectral_flatness = features.get("spectral_flatness", 0.0)
        if spectral_flatness > 0.05:
            scores.append(0.72)
            anomaly_reasons.append("Acoustically unnatural noise distribution in spectral envelope")
        else:
            scores.append(0.20)

        # Compute raw weighted mean score for Layer 1
        raw_score = float(np.mean(scores))
        
        # Clamp score within [0.05, 0.95] for probabilistic output
        layer1_score = max(0.05, min(0.95, round(raw_score, 4)))

        details = {
            "status": "configured",
            "score": layer1_score,
            "anomalies": anomaly_reasons,
            "features_summary": {
                "pitch_mean_hz": round(features.get("pitch_mean", 0.0), 2),
                "pitch_std": round(pitch_std, 2),
                "f0_jitter": round(f0_jitter, 5),
                "spectral_centroid": round(features.get("centroid_mean", 0.0), 2),
                "spectral_bandwidth": round(bandwidth_mean, 2),
                "zcr": round(features.get("zcr_mean", 0.0), 4),
                "rms_energy": round(features.get("rms_mean", 0.0), 4),
                "spectral_flatness": round(spectral_flatness, 6)
            }
        }

        return layer1_score, details
