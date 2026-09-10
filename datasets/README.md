# Datasets Guide for VoiceShield Evaluation & Training

To train and evaluate VoiceShield on real-world human vs AI deepfake audio, you can use public audio anti-spoofing benchmark datasets.

---

## 1. Recommended Public Benchmark Datasets

### A. ASVspoof 2019 / 2021 (Logical Access - LA)
- **Source**: [https://www.asvspoof.org/](https://www.asvspoof.org/)
- **Description**: The industry standard benchmark dataset for synthetic speech and voice conversion detection.
- **Content**: Thousands of genuine human speech recordings and synthetic audio generated using state-of-the-art TTS (Neural Vocoders, Tacotron, WaveNet, VITS) and Voice Conversion algorithms.

### B. In-the-Wild Audio Deepfake Dataset
- **Source**: [https://deepfake-demo.xrai.56kb.org/](https://deepfake-demo.xrai.56kb.org/)
- **Description**: Real-world deepfake audio samples harvested from YouTube, social media, podcasts, and public figures.

### C. ASVspoof 5 (Latest 2024 Benchmark)
- **Source**: [https://www.asvspoof.org/asvspoof2024](https://www.asvspoof.org/asvspoof2024)

---

## 2. Directory Structure for Training & Evaluation

Organize dataset audio files into `real` and `fake` subdirectories:

```
datasets/
├── real/
│   ├── human_01.wav
│   ├── human_02.wav
│   └── ...
└── fake/
    ├── synthetic_01.wav
    ├── synthetic_02.wav
    └── ...
```

---

## 3. Generating Sample Synthetic Test Dataset

For immediate testing, offline evaluation, and model training without downloading large datasets, run the included sample generator script:

```bash
python datasets/generate_samples.py
```
This populates `datasets/real/` and `datasets/fake/` with synthetic test audio files.
