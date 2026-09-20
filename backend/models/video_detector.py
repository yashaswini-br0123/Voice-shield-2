import os
import cv2
import tempfile
import numpy as np
from typing import Dict, Any, Tuple
from backend.models.image_detector import ImageDetector
from backend.models.acoustic_detector import AcousticDetector
from backend.models.aasist_detector import AASISTDetector
from backend.models.spectral_detector import SpectralDetector
from backend.ensemble.fusion import EnsembleFusion
from backend.audio.preprocessing import load_and_preprocess_audio


class VideoDetector:
    """
    Video Deepfake & Multimodal Detection Module.
    Combines:
    1. Visual Frame-by-Frame Analysis (2D FFT, ELA, Facial Blur Consistency across keyframes)
    2. Audio Track Deepfake Analysis (3-Layer Acoustic, Waveform AASIST, Spectral LFCC)
    3. Audio-Visual Multimodal Score Fusion
    """
    def __init__(self):
        self._image_detector = None
        self._acoustic_detector = None
        self._aasist_detector = None
        self._spectral_detector = None
        self._fusion = None

    @property
    def image_detector(self):
        if self._image_detector is None:
            self._image_detector = ImageDetector()
        return self._image_detector

    @property
    def acoustic_detector(self):
        if self._acoustic_detector is None:
            self._acoustic_detector = AcousticDetector()
        return self._acoustic_detector

    @property
    def aasist_detector(self):
        if self._aasist_detector is None:
            self._aasist_detector = AASISTDetector()
        return self._aasist_detector

    @property
    def spectral_detector(self):
        if self._spectral_detector is None:
            self._spectral_detector = SpectralDetector()
        return self._spectral_detector

    @property
    def fusion(self):
        if self._fusion is None:
            self._fusion = EnsembleFusion()
        return self._fusion

    def analyze(self, video_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Extracts keyframes and audio track from video file, runs visual and acoustic detection pipelines,
        and computes multimodal Video Deepfake Probability.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.5, {
                "status": "error",
                "error": "Failed to open video file.",
                "score": 0.5
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # 1. Fast Keyframe Sampling (4 uniform keyframes in RAM)
        sample_count = 4 if total_frames >= 4 else max(1, total_frames)
        frame_indices = np.linspace(0, max(0, total_frames - 1), sample_count, dtype=int)
        
        frame_scores = []
        frame_anomalies = []

        try:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Run fast in-memory image analysis on frame
                score, details = self.image_detector.analyze_cv_image(frame)
                frame_scores.append(score)
                for anom in details.get("anomalies", []):
                    if anom and anom not in frame_anomalies:
                        frame_anomalies.append(anom)
        finally:
            cap.release()

        visual_ai_prob = float(np.mean(frame_scores)) if frame_scores else 0.5

        # 2. Audio Track Deepfake Analysis
        has_audio = False
        audio_details = {}
        audio_ai_prob = 0.5

        try:
            # Load and preprocess audio from video file using pydub / librosa
            audio_array, audio_meta = load_and_preprocess_audio(video_path)
            
            s1, d1 = self.acoustic_detector.analyze(audio_array, sr=audio_meta["sample_rate"])
            s2, d2 = self.aasist_detector.analyze(audio_array, sr=audio_meta["sample_rate"])
            s3, d3 = self.spectral_detector.analyze(audio_array, sr=audio_meta["sample_rate"])
            
            ens_res = self.fusion.fuse_scores(d1, d2, d3)
            audio_ai_prob = ens_res["ai_probability"]
            audio_details = ens_res
            has_audio = True
        except Exception:
            # Video has no audio track or audio extraction failed
            has_audio = False

        # 3. Multimodal Score Fusion
        if has_audio:
            # Weight Visual (55%) + Audio (45%)
            final_score = (0.55 * visual_ai_prob) + (0.45 * audio_ai_prob)
        else:
            final_score = visual_ai_prob

        final_score = max(0.05, min(0.95, round(final_score, 4)))

        # Collect overall anomalies
        all_anomalies = frame_anomalies
        if has_audio and "all_anomalies" in audio_details:
            for anom in audio_details["all_anomalies"]:
                if anom and anom not in all_anomalies:
                    all_anomalies.append(anom)

        return final_score, {
            "status": "configured",
            "score": final_score,
            "visual_ai_prob": round(visual_ai_prob, 3),
            "audio_ai_prob": round(audio_ai_prob, 3) if has_audio else None,
            "has_audio_track": has_audio,
            "duration_sec": round(duration_sec, 2),
            "frames_analyzed": len(frame_scores),
            "anomalies": all_anomalies
        }
