import os
import numpy as np
from typing import Dict, Any, Tuple
from backend.config import settings

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    torchaudio = None


class WavLMDetector:
    """
    Layer 3: WavLM Self-Supervised Speech Representation Module
    Uses torchaudio.pipelines.WAVLM_BASE to extract 768-dimensional contextual speech representations.
    
    Strict Rule:
    - If a trained downstream deepfake classifier checkpoint exists (settings.WAVLM_CLASSIFIER_PATH),
      it evaluates the representation vector and returns a calibrated AI Risk Score.
    - If no trained downstream classifier checkpoint exists, it returns representation profiling
      ('status': 'representation_profiling', 'score': None) without fabricating arbitrary fake scores or weights.
    """
    def __init__(self):
        self.model_path = getattr(settings, "WAVLM_CLASSIFIER_PATH", os.path.join(settings.BASE_DIR, "model_weights", "wavlm_classifier.pth"))
        self.ssl_model = None
        self.classifier = None
        self.is_ssl_loaded = False
        self.is_classifier_loaded = False
        self._load_models()

    def _load_models(self):
        if HAS_TORCH and torchaudio is not None:
            try:
                bundle = torchaudio.pipelines.WAVLM_BASE
                self.ssl_model = bundle.get_model()
                self.ssl_model.eval()
                self.is_ssl_loaded = True
            except Exception as e:
                print(f"[WavLMDetector] SSL model load warning: {e}")
                self.is_ssl_loaded = False

            if os.path.exists(self.model_path):
                try:
                    self.classifier = torch.load(self.model_path, map_location=torch.device('cpu'))
                    self.classifier.eval()
                    self.is_classifier_loaded = True
                except Exception as e:
                    print(f"[WavLMDetector] Classifier checkpoint load warning: {e}")
                    self.is_classifier_loaded = False

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extracts 768-dimensional summary representation vector (mean across time frames).
        """
        if not self.is_ssl_loaded or self.ssl_model is None:
            return np.zeros(768, dtype=np.float32)

        try:
            tensor_in = torch.from_numpy(audio).float().unsqueeze(0) # [1, num_samples]
            with torch.no_grad():
                features, _ = self.ssl_model(tensor_in)
                # features shape: [1, num_frames, 768]
                emb = features[0].mean(dim=0).cpu().numpy() # [768]
                return emb.astype(np.float32)
        except Exception as e:
            print(f"[WavLMDetector] Feature extraction error: {e}")
            return np.zeros(768, dtype=np.float32)

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Tuple[float, Dict[str, Any]]:
        """
        Analyzes audio via WavLM SSL model.
        Returns score (float or None) and technical evidence breakdown.
        """
        emb = self.extract_features(audio, sr=sr)
        emb_norm = float(np.linalg.norm(emb))

        if self.is_classifier_loaded and self.classifier is not None:
            try:
                tensor_emb = torch.from_numpy(emb).float().unsqueeze(0)
                with torch.no_grad():
                    logits = self.classifier(tensor_emb)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    # Index 1 = Spoof (AI) if binary classifier
                    ai_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    ai_score = max(0.05, min(0.95, round(ai_score, 4)))
                
                return ai_score, {
                    "status": "configured",
                    "score": ai_score,
                    "model": "WavLM Base SSL Encoder + Trained Classifier Head",
                    "embedding_dim": 768,
                    "embedding_norm": round(emb_norm, 4),
                    "checkpoint_loaded": True,
                    "anomalies": [
                        "WavLM SSL contextual embedding indicates synthetic vocoder acoustic footprint" if ai_score > 0.6 else "WavLM SSL contextual representation matches natural speech dynamics"
                    ]
                }
            except Exception as e:
                print(f"[WavLMDetector] Classifier inference error: {e}")

        # Representation Profiling Mode (No trained classifier head)
        return None, {
            "status": "representation_profiling",
            "score": None,
            "model": "WavLM Base SSL Feature Extractor (torchaudio.pipelines.WAVLM_BASE)",
            "embedding_dim": 768,
            "embedding_norm": round(emb_norm, 4),
            "checkpoint_loaded": False,
            "message": f"WavLM SSL 768-dim embeddings extracted (norm={emb_norm:.2f}). Downstream classifier head checkpoint unconfigured at '{self.model_path}'.",
            "anomalies": [
                "WavLM SSL representations extracted (768-dim vector). Profiling active without fake score assignment."
            ]
        }
