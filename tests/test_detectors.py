import numpy as np
import pytest
from backend.models.acoustic_detector import AcousticDetector
from backend.models.aasist_detector import AASISTDetector
from backend.models.spectral_detector import SpectralDetector
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


def test_ensemble_fusion():
    fusion = EnsembleFusion(decision_thresh=0.50)
    
    l1 = {"status": "configured", "score": 0.85, "anomalies": ["Pitch rigid"]}
    l2 = {"status": "configured", "score": 0.90, "anomalies": ["Phase distortion"]}
    l3 = {"status": "configured", "score": 0.80, "anomalies": ["LFCC anomaly"]}

    result = fusion.fuse_scores(l1, l2, l3)
    assert result["prediction"] == "Likely AI-Generated"
    assert result["ai_probability"] > 0.50
    assert len(result["all_anomalies"]) > 0
