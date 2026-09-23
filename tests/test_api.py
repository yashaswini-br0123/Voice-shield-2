import io
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile
import numpy as np
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "demo_mode" in data
    assert "models" in data


def test_analyze_endpoint():
    # Create sample WAV bytes
    sr = 16000
    t = np.linspace(0, 1.0, sr)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, signal)
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        files={"file": ("test_voice.wav", buf, "audio/wav")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "ai_probability" in data
    assert "confidence" in data
    assert "layers" in data
    assert "audio" in data
    assert data["audio"]["sample_rate"] == 16000


def test_analyze_url_endpoint_empty_url():
    response = client.post(
        "/api/analyze_url",
        json={"url": ""}
    )
    assert response.status_code == 400
    assert "detail" in response.json()


def test_analyze_url_endpoint_invalid_stream():
    response = client.post(
        "/api/analyze_url",
        json={"url": "https://invalid-nonexistent-domain-xyz123.com/fake.mp4"}
    )
    assert response.status_code == 400
    assert "detail" in response.json()
