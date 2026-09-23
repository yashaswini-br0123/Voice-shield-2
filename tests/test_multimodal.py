import os
import tempfile
import numpy as np
import pytest
from PIL import Image
from backend.models.image_detector import ImageDetector
from backend.models.video_detector import VideoDetector


def test_specxnet_image_detector():
    # Create sample synthetic RGB test image
    img_data = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_data)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_img.save(tmp_path, "JPEG")

    try:
        detector = ImageDetector()
        score, details = detector.analyze(tmp_path)
        assert 0.0 <= score <= 1.0
        assert details["status"] == "configured"
        assert details["detector"] == "SpecXNet Dual-Domain Architecture"
        assert "fft_spectral_score" in details
        assert "ela_compression_score" in details
        assert "heatmap_url" in details
        assert isinstance(details["heatmap_url"], str)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fakestormer_video_detector():
    detector = VideoDetector()
    # Test gracefully returning baseline score when non-existent video is passed
    score, details = detector.analyze("non_existent_video.mp4")
    assert 0.0 <= score <= 1.0
    assert details["status"] in ["warning", "error"]
    assert details["detector"] == "FakeSTormer Spatio-Temporal Video Deepfake Detector" or "error" in details
