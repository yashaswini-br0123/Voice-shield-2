import os
import tempfile
import numpy as np
import pytest
from scipy.io import wavfile
from backend.audio.preprocessing import load_and_preprocess_audio
from backend.audio.feature_extraction import extract_acoustic_features, extract_lfcc_features


def test_audio_preprocessing_wav():
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        wavfile.write(tmp_path, sr, (signal * 32767).astype(np.int16))

    try:
        audio, meta = load_and_preprocess_audio(tmp_path)
        assert len(audio) > 0
        assert meta["sample_rate"] == 16000
        assert meta["duration"] >= 0.9
        assert meta["channels"] == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_feature_extraction():
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    signal = 0.5 * np.sin(2 * np.pi * 220 * t)

    ac_feats = extract_acoustic_features(signal, sr)
    assert "mfcc_mean" in ac_feats
    assert "centroid_mean" in ac_feats

    lfcc_feats = extract_lfcc_features(signal, sr)
    assert isinstance(lfcc_feats, np.ndarray)
    assert len(lfcc_feats) == 80 # 20 mean + 20 std + 20 delta_mean + 20 delta_std
