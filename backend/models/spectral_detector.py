import os
try:
    import joblib
except Exception:
    joblib = None
import numpy as np
from typing import Dict, Any, Tuple
from backend.audio.feature_extraction import extract_lfcc_features
from backend.config import settings


class SpectralDetector:
    """
    Layer 3: Spectral Analysis Module
    Extracts LFCC (Linear Frequency Cepstral Coefficients) features and evaluates synthetic voice
    likelihood using a trained Scikit-Learn Machine Learning Classifier (Random Forest / Logistic Regression).
    """
    def __init__(self):
        self.model_path = settings.SPECTRAL_MODEL_PATH
        self.classifier = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if joblib is not None and os.path.exists(self.model_path):
            try:
                self.classifier = joblib.load(self.model_path)
                self.is_loaded = True
            except Exception as e:
                print(f"[SpectralDetector] Error loading model '{self.model_path}': {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Tuple[float, Dict[str, Any]]:
        """
        Extracts 80-dimensional LFCC feature vector and predicts synthetic probability.
        """
        # Active speech extraction
        active_speech = audio[np.abs(audio) > 0.02]
        if len(active_speech) < 1600:
            active_speech = audio

        lfcc_vec = extract_lfcc_features(active_speech, sr=sr)
        
        if self.is_loaded and self.classifier is not None:
            try:
                # Shape: [1, 80]
                X = lfcc_vec.reshape(1, -1)
                probs = self.classifier.predict_proba(X)[0]
                # Index 1 is synthetic/fake probability
                score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                score = max(0.05, min(0.95, round(score, 4)))
                
                return score, {
                    "status": "configured",
                    "score": score,
                    "model": "LFCC + Scikit-Learn Classifier (Random Forest / Logistic Regression)",
                    "num_features": len(lfcc_vec),
                    "anomalies": [
                        "LFCC spectral envelope distortion detected in linear frequency filterbanks" if score > 0.6 else "LFCC cepstral coefficients match natural vocal tract resonances"
                    ]
                }
            except Exception as e:
                print(f"[SpectralDetector] Classifier inference error: {e}")

        # Baseline heuristic calculation using LFCC feature vector formant variance
        lfcc_std = lfcc_vec[20:40]
        lfcc_std_mean = float(np.mean(lfcc_std)) if len(lfcc_std) > 0 else 0.0
        
        if lfcc_std_mean < 0.80:
            score = 0.78
            anomaly = "Linear cepstral sub-band energy shows static frame-to-frame formant variance (synthetic vocoder signature)"
        else:
            score = 0.18
            anomaly = "LFCC features display natural frame-to-frame dynamic vocal tract resonance variance"

        status_label = "demo_mode" if settings.DEMO_MODE else "baseline_model"
        
        return score, {
            "status": status_label,
            "score": score,
            "model": "LFCC Baseline Detector (Run train_spectral.py to build custom model)",
            "num_features": len(lfcc_vec),
            "anomalies": [anomaly]
        }
