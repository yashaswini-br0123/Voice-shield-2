# VoiceShield Model Weights & Checkpoints Directory

This directory stores pretrained machine learning model checkpoints for VoiceShield.

## Pretrained Models Structure

```
model_weights/
├── aasist.pth           # PyTorch AASIST raw waveform deepfake detector model weights
└── spectral_clf.joblib  # Trained Scikit-Learn classifier for Layer 3 LFCC analysis
```

---

## 1. Layer 2: AASIST Checkpoint Setup (`aasist.pth`)

**AASIST** (*Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Neural Networks*) is an end-to-end raw waveform spoofing detector.

### How to Download Pretrained Weights:
1. Download official AASIST pretrained weights from the ASVspoof / AASIST GitHub Repository:
   - Official Repo: [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)
   - Direct Weights Link: `AASIST.pth` trained on ASVspoof2019 LA benchmark.
2. Rename the downloaded file to `aasist.pth` and place it in this directory:
   ```
   model_weights/aasist.pth
   ```

### Status when missing:
If `aasist.pth` is not placed here:
- When `DEMO_MODE=false`, VoiceShield will mark Layer 2 as `unconfigured` and dynamically re-weight Layer 1 and Layer 3 in the ensemble.
- When `DEMO_MODE=true`, Layer 2 will display interactive presentation output with an explicit notification banner.

---

## 2. Layer 3: Spectral Classifier (`spectral_clf.joblib`)

Layer 3 uses Linear Frequency Cepstral Coefficients (LFCC) + Scikit-Learn classifier (`RandomForestClassifier` or `LogisticRegression`).

### How to Train:
Run the built-in training script on your audio dataset:
```bash
python train_spectral.py --dataset_dir datasets/
```
This will automatically generate `model_weights/spectral_clf.joblib`.
