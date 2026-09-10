import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Dict, Any, Tuple
from backend.config import settings


class ImageDetector:
    """
    Image Deepfake & AI Generation Analysis Module.
    Analyzes:
    1. 2D Frequency Spectrum Artifacts (FFT 2D spectral energy rings)
    2. Error Level Analysis (ELA) JPEG compression residual variance
    3. Color Channel Noise & Covariance
    4. Facial Edge & Gradient Blur Consistency
    """
    def __init__(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.face_cascade = None
        except Exception:
            self.face_cascade = None

    def analyze(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Loads an image file, extracts spatial, spectral, and compression features,
        and returns AI-Generated probability [0.0 - 1.0] and technical analysis details.
        """
        try:
            pil_img = Image.open(image_path).convert('RGB')
            # Convert PIL RGB image to OpenCV BGR numpy array to prevent OpenCV Unicode path read failures
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return 0.5, {
                "status": "error",
                "error": f"Failed to load image: {str(e)}",
                "score": 0.5,
                "anomalies": [f"Image processing notice: {str(e)}"]
            }

        scores = []
        anomalies = []

        # 1. 2D FFT High-Frequency Spectral Ring Analysis
        fft_score, fft_anom = self._analyze_fft_spectrum(cv_img)
        scores.append(fft_score)
        if fft_anom:
            anomalies.append(fft_anom)

        # 2. Error Level Analysis (ELA)
        ela_score, ela_anom = self._analyze_ela(pil_img, image_path)
        scores.append(ela_score)
        if ela_anom:
            anomalies.append(ela_anom)

        # 3. Color Channel Correlation & Spatial Noise Variance
        noise_score, noise_anom = self._analyze_noise_covariance(cv_img)
        scores.append(noise_score)
        if noise_anom:
            anomalies.append(noise_anom)

        # 4. Facial Edge & Blur Consistency
        face_score, face_anom, num_faces = self._analyze_facial_gradients(cv_img)
        if num_faces > 0:
            scores.append(face_score)
            if face_anom:
                anomalies.append(face_anom)

        raw_score = float(np.mean(scores))
        ai_prob = max(0.05, min(0.95, round(raw_score, 4)))

        width, height = pil_img.size

        return ai_prob, {
            "status": "configured",
            "score": ai_prob,
            "anomalies": anomalies,
            "dimensions": f"{width}x{height}",
            "faces_detected": num_faces,
            "fft_spectral_score": round(fft_score, 3),
            "ela_compression_score": round(ela_score, 3),
            "noise_covariance_score": round(noise_score, 3)
        }

    def analyze_cv_image(self, cv_img: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Fast in-memory analysis directly on OpenCV BGR numpy array (zero disk I/O)."""
        scores = []
        anomalies = []

        fft_score, fft_anom = self._analyze_fft_spectrum(cv_img)
        scores.append(fft_score)
        if fft_anom: anomalies.append(fft_anom)

        noise_score, noise_anom = self._analyze_noise_covariance(cv_img)
        scores.append(noise_score)
        if noise_anom: anomalies.append(noise_anom)

        face_score, face_anom, num_faces = self._analyze_facial_gradients(cv_img)
        if num_faces > 0:
            scores.append(face_score)
            if face_anom: anomalies.append(face_anom)

        raw_score = float(np.mean(scores))
        ai_prob = max(0.05, min(0.95, round(raw_score, 4)))

        h, w = cv_img.shape[:2]

        return ai_prob, {
            "status": "configured",
            "score": ai_prob,
            "anomalies": anomalies,
            "dimensions": f"{w}x{h}",
            "faces_detected": num_faces,
            "fft_spectral_score": round(fft_score, 3),
            "ela_compression_score": 0.5,
            "noise_covariance_score": round(noise_score, 3)
        }

    def _analyze_fft_spectrum(self, cv_img: np.ndarray) -> Tuple[float, str]:
        """Analyzes 2D FFT frequency spectrum for grid artifacts common in GANs/Diffusion."""
        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6)

            h, w = magnitude_spectrum.shape
            cy, cx = h // 2, w // 2
            radius = 120
            
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            center_energy = np.mean(magnitude_spectrum[mask])
            outer_energy = np.mean(magnitude_spectrum[~mask])
            
            ratio = outer_energy / (center_energy + 1e-6)

            if ratio > 0.78:
                return 0.81, "2D FFT frequency spectrum exhibits artificial high-frequency grid artifacts (diffusion/GAN signature)"
            elif ratio < 0.15:
                return 0.72, "Oversmoothed high-frequency spectral roll-off typical of AI image generators"
            else:
                return 0.20, ""
        except Exception:
            return 0.22, ""

    def _analyze_ela(self, pil_img: Image.Image, file_path: str) -> Tuple[float, str]:
        """Error Level Analysis (ELA) JPEG compression residual check."""
        try:
            ela_temp = file_path + "_ela_tmp.jpg"
            pil_img.save(ela_temp, 'JPEG', quality=90)
            recompressed = Image.open(ela_temp)
            
            diff = ImageChops.difference(pil_img, recompressed)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
                
            scale = 255.0 / max_diff
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            
            diff_np = np.array(diff)
            ela_std = float(np.std(diff_np))
            
            if os.path.exists(ela_temp):
                try:
                    os.remove(ela_temp)
                except Exception:
                    pass

            if ela_std > 68.0:
                return 0.79, "ELA (Error Level Analysis) reveals inconsistent JPEG compression residual variance across pixel regions"
            else:
                return 0.22, ""
        except Exception:
            return 0.22, ""

    def _analyze_noise_covariance(self, cv_img: np.ndarray) -> Tuple[float, str]:
        """Evaluates color channel noise covariance and pixel smoothness."""
        try:
            b, g, r = cv2.split(cv_img)
            var_b = cv2.Laplacian(b, cv2.CV_64F).var()
            var_g = cv2.Laplacian(g, cv2.CV_64F).var()
            var_r = cv2.Laplacian(r, cv2.CV_64F).var()
            
            lap_mean = (var_b + var_g + var_r) / 3.0

            if lap_mean < 45.0:
                return 0.78, "Oversmoothed pixel texture with missing natural camera sensor noise"
            elif abs(var_b - var_r) < 1.0 and lap_mean > 800:
                return 0.74, "Synthetic color channel noise correlation across RGB components"
            else:
                return 0.20, ""
        except Exception:
            return 0.20, ""

    def _analyze_facial_gradients(self, cv_img: np.ndarray) -> Tuple[float, str, int]:
        """Detects faces and evaluates edge sharpness consistency around facial boundaries."""
        if self.face_cascade is None:
            return 0.22, "", 0

        try:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            num_faces = len(faces)

            if num_faces == 0:
                return 0.22, "", 0

            face_vars = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_vars.append(cv2.Laplacian(face_roi, cv2.CV_64F).var())

            bg_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            avg_face_var = np.mean(face_vars)

            ratio = avg_face_var / (bg_var + 1e-6)
            if ratio > 8.0 or ratio < 0.12:
                return 0.82, "Facial edge gradient mismatch detected relative to background (Face Deepfake artifact)", num_faces
            else:
                return 0.20, "", num_faces
        except Exception:
            return 0.22, "", 0
