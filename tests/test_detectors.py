import numpy as np
import pytest
from backend.models.acoustic_detector import AcousticDetector
from backend.models.aasist_detector import AASISTDetector
from backend.models.spectral_detector import SpectralDetector
from backend.models.wavlm_detector import WavLMDetector
from backend.ensemble.fusion import EnsembleFusion


def test_acoustic_detector():
    audio = np.random.normal(0, 0.1, 16000 * 2)
    detector = AcousticDetector()
    score, details = detector.analyze(audio, sr=16000)
    assert 0.0 <= score <= 1.0
    assert "status" in details
    assert "features_summary" in details


def test_aasist_detector():
    audio = np.random.normal(0, 0.1, 16000 * 2)
    detector = AASISTDetector()
    score, details = detector.analyze(audio, sr=16000)
    assert "status" in details
    assert details["status"] in ["configured", "unconfigured", "demo_mode"]


def test_spectral_detector():
    audio = np.random.normal(0, 0.1, 16000 * 2)
    detector = SpectralDetector()
    score, details = detector.analyze(audio, sr=16000)
    assert 0.0 <= score <= 1.0
    assert "status" in details


def test_wavlm_detector():
    audio = np.random.normal(0, 0.1, 16000 * 2)
    detector = WavLMDetector()
    score, details = detector.analyze(audio, sr=16000)
    assert "status" in details
    assert details["status"] in ["configured", "representation_profiling"]
    assert "embedding_dim" in details
    assert details["embedding_dim"] == 768


def test_ensemble_fusion():
    fusion = EnsembleFusion(human_thresh=0.35, ai_thresh=0.65, disagreement_thresh=0.28)
    
    l1 = {"status": "configured", "score": 0.85, "anomalies": ["Pitch rigid"]}
    l2 = {"status": "configured", "score": 0.90, "anomalies": ["Phase distortion"]}
    l3 = {"status": "configured", "score": 0.80, "anomalies": ["LFCC anomaly"]}
    l4 = {"status": "representation_profiling", "score": None, "anomalies": ["WavLM SSL 768-dim vector"]}

    result = fusion.fuse_scores(l1, l2, l3, l4)
    assert result["prediction"] == "Likely AI-Generated"
    assert result["ai_risk_score"] > 0.65
    assert len(result["all_anomalies"]) > 0
