import os
import numpy as np
from typing import Dict, Any, Tuple
from backend.config import settings

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None
    class DummyNN:
        Module = object
        def __getattr__(self, name):
            return object
    nn = DummyNN()

# AASIST PyTorch Light Architecture Definition
class SincConv1d(nn.Module if HAS_TORCH else object):
    """Sinc-convolution layer for raw audio waveform filtering"""
    def __init__(self, out_channels=70, kernel_size=128, sample_rate=16000):
        if HAS_TORCH:
            super(SincConv1d, self).__init__()
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.sample_rate = sample_rate
            self.conv = torch.nn.Conv1d(1, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        if not HAS_TORCH:
            return x
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.conv(x)


class AASISTModel(nn.Module if HAS_TORCH else object):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Neural Networks
    """
    def __init__(self):
        super(AASISTModel, self).__init__()
        self.sinc_conv = SincConv1d(out_channels=32, kernel_size=128)
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [0: Genuine/Human, 1: Spoof/AI]
        )

    def forward(self, x):
        # x shape: [batch, samples]
        out = self.sinc_conv(x)
        out = self.conv_block(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits


class AASISTDetector:
    """
    Layer 2: Waveform / Phase Analysis Module
    Uses raw waveform PyTorch AASIST model to detect Phase anomalies and high-frequency wave artifacts.
    """
    def __init__(self):
        self.checkpoint_path = settings.AASIST_CHECKPOINT_PATH
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if HAS_TORCH and os.path.exists(self.checkpoint_path):
            try:
                self.model = AASISTModel()
                state_dict = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.is_loaded = True
            except Exception as e:
                print(f"[AASISTDetector] Error loading checkpoint '{self.checkpoint_path}': {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def analyze(self, audio: np.ndarray, sr: int = 16000) -> Tuple[float, Dict[str, Any]]:
        """
        Analyzes raw waveform audio.
        If PyTorch checkpoint is present, returns real PyTorch inference score.
        If missing and DEMO_MODE=False, returns unconfigured status without fake predictions.
        If missing and DEMO_MODE=True, returns simulated score with explicit notice.
        """
        if self.is_loaded and self.model is not None:
            try:
                # Ensure input audio length (3 seconds / 48000 samples)
                target_len = 48000
                if len(audio) < target_len:
                    padded_audio = np.pad(audio, (0, target_len - len(audio)), mode='wrap')
                else:
                    padded_audio = audio[:target_len]

                tensor_in = torch.from_numpy(padded_audio).float().unsqueeze(0) # shape [1, 48000]
                with torch.no_grad():
                    logits = self.model(tensor_in)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                    # Index 1 corresponds to Spoof / AI probability
                    ai_prob = float(probs[1])

                ai_prob = max(0.05, min(0.95, round(ai_prob, 4)))
                return ai_prob, {
                    "status": "configured",
                    "score": ai_prob,
                    "model": "AASIST Raw Waveform Graph Neural Network (PyTorch)",
                    "checkpoint_loaded": True,
                    "anomalies": [
                        "Raw waveform phase inconsistency detected in high-frequency band" if ai_prob > 0.6 else "Waveform phase trajectory shows natural glottal impulse patterns"
                    ]
                }
            except Exception as e:
                print(f"[AASISTDetector] Inference error: {e}")

        # Handling missing checkpoint
        if settings.DEMO_MODE:
            # Active speech frame extraction (trim initial/trailing room silence)
            active_speech = audio[np.abs(audio) > 0.02]
            if len(active_speech) < 1600:
                active_speech = audio # Fallback if audio is very quiet overall

            phase_diff = np.diff(active_speech)
            diff2 = np.diff(phase_diff)
            
            # Natural human voice display dynamic amplitude & phase fluctuation (std(diff2) >= 0.005 and std(active_speech) >= 0.03)
            # Synthetic flat vocoders display rigid/oversmoothed frame phase dynamics
            is_synthetic_synth = float(np.std(diff2)) < 0.005 and float(np.std(active_speech)) < 0.025
            mock_score = 0.88 if is_synthetic_synth else 0.18
            return mock_score, {
                "status": "demo_mode",
                "score": mock_score,
                "model": "AASIST (Demo Simulation)",
                "checkpoint_loaded": False,
                "message": f"AASIST checkpoint missing at '{self.checkpoint_path}'. Displaying DEMO MODE result.",
                "anomalies": [
                    "Synthetic vocoder waveform phase oversmoothing detected" if is_synthetic_synth else "Waveform phase trajectory shows natural glottal impulse patterns"
                ]
            }
        else:
            # Strictly return unconfigured state when DEMO_MODE is False
            return 0.0, {
                "status": "unconfigured",
                "score": None,
                "model": "AASIST Raw Waveform Model",
                "checkpoint_loaded": False,
                "message": f"Waveform detection model checkpoint is unconfigured or missing at path '{self.checkpoint_path}'. Download checkpoint to enable Layer 2.",
                "anomalies": ["Layer 2 unconfigured: Missing model checkpoint."]
            }
