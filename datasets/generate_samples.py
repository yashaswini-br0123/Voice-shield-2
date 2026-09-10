import os
import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile


def generate_all_samples(base_dir="datasets"):
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # 1. Generate Human Audio
    f0 = 120.0 + 15.0 * np.sin(2 * np.pi * 4.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    audio_human = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase) + np.random.normal(0, 0.02, len(t))
    audio_human = audio_human / np.max(np.abs(audio_human))
    wavfile.write(os.path.join(real_dir, "human_sample_01.wav"), sample_rate, (audio_human * 32767).astype(np.int16))

    # 2. Generate AI Audio
    f0_ai = 140.0 # Rigid F0
    phase_ai = 2 * np.pi * f0_ai * t
    audio_ai = 0.6 * np.sin(phase_ai) + 0.3 * np.sin(2 * phase_ai)
    spec = np.fft.rfft(audio_ai)
    freqs = np.fft.rfftfreq(len(audio_ai), 1 / sample_rate)
    spec[freqs > 7000] = 0.0 # High freq cutoff
    audio_ai = np.fft.irfft(spec)
    audio_ai = audio_ai / np.max(np.abs(audio_ai))
    wavfile.write(os.path.join(fake_dir, "ai_sample_01.wav"), sample_rate, (audio_ai * 32767).astype(np.int16))

    # 3. Generate Real Image Sample (natural noise & smooth gradients)
    img_real = np.zeros((400, 400, 3), dtype=np.uint8)
    for y in range(400):
        for x in range(400):
            img_real[y, x] = [int(100 + 100 * (x/400)), int(120 + 80 * (y/400)), 150]
    # Add natural camera sensor noise
    noise = np.random.normal(0, 8, (400, 400, 3)).astype(np.int16)
    img_real = np.clip(img_real.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(real_dir, "real_image_sample.jpg"), img_real)

    # 4. Generate AI Image Sample (with artificial high-frequency FFT grid artifacts)
    img_ai = np.zeros((400, 400, 3), dtype=np.uint8)
    y_grid, x_grid = np.ogrid[:400, :400]
    grid_pattern = (np.sin(x_grid * 0.4) * np.cos(y_grid * 0.4) * 40).astype(np.uint8)
    img_ai[:, :, 0] = np.clip(120 + grid_pattern, 0, 255)
    img_ai[:, :, 1] = np.clip(100 + grid_pattern, 0, 255)
    img_ai[:, :, 2] = np.clip(140 + grid_pattern, 0, 255)
    cv2.imwrite(os.path.join(fake_dir, "ai_image_sample.jpg"), img_ai)

    print("Sample generation complete (Audio & Image)!")


if __name__ == "__main__":
    generate_all_samples()
