# VoiceShield – AI Voice Deepfake Detection & Prevention Platform

> **VoiceShield** is a multi-layer web application and machine learning engine designed to evaluate whether a voice recording is **HUMAN** or **AI-GENERATED / SYNTHETIC**.

---

## ⚠️ Important Scientific & Probabilistic Disclaimer
VoiceShield performs **probabilistic detection** based on multi-layer spectro-temporal audio feature extraction. It **does NOT claim 100% accuracy or infallible results**. Modern AI voice clones (e.g. neural vocoders, voice conversion models) evolve continuously. Results are confidence scores meant for risk assessment and auditing.

---

## 📐 System Architecture

```
                       User Audio Input
       (WAV, MP3, M4A, FLAC, OGG, WebM, Browser Microphone)
                              │
                              ▼
                      Audio Preprocessing
            (Standardization to 16kHz Mono WAV,
              Peak Normalization, Duration Check)
                              │
         ┌────────────────────┴────────────────────┐
         │          3-LAYER DETECTION SYSTEM        │
         │                                         │
         │ Layer 1: Acoustic Analysis              │
         │   • MFCC, Mel-Spectrogram, Pitch/F0     │
         │   • Spectral Centroid, Bandwidth, ZCR   │
         │   • Reverberation / Room Acoustics      │
         │                                         │
         │ Layer 2: Waveform & Phase Analysis      │
         │   • AASIST Raw Waveform PyTorch Model   │
         │   • SincConv Filterbank & GNN           │
         │                                         │
         │ Layer 3: Spectral LFCC Analysis         │
         │   • Linear Frequency Cepstral Coeffs    │
         │   • Trained ML Classifier (Scikit-Learn)│
         └────────────────────┬────────────────────┘
                              │
                              ▼
                    Ensemble Score Fusion
         (Dynamic Weighting & Softmax Score Fusion)
                              │
                              ▼
                      Final Prediction
           ┌─────────────────────────────────────┐
           │ • Verdict: "Likely AI-Generated"    │
           │            "Likely Human"           │
           │            "Uncertain"              │
           │ • Confidence Score                  │
           │ • 3-Layer Breakdown                 │
           │ • Technical Explanation & Anomalies │
           └─────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, Modern Cyber-Dark Dashboard, JavaScript (ES6+), Web Audio API, MediaRecorder API, HTML5 Canvas Visualizer.
- **Backend**: Python 3.10+, FastAPI, Uvicorn, REST API, Pydantic v2.
- **Machine Learning & Audio Processing**: PyTorch, torchaudio, librosa, NumPy, SciPy, scikit-learn, soundfile, joblib.
- **Security & Privacy**: Automatic temporary file cleanup, input file validation, MIME verification, rate-limiting middleware, CORS protection.

---

## 🚀 Quick Start & Installation Guide

### 1. Prerequisites
- Python 3.10 or higher installed.
- **FFmpeg** installed on system PATH (required for processing MP3, M4A, FLAC, AAC files via pydub/librosa).
  - *Windows*: Download from [FFmpeg.org](https://ffmpeg.org/) or `winget install FFmpeg`

### 2. Clone Repository & Setup Environment
```bash
# Navigate to project root directory
cd "voice shield 2"

# Create Python Virtual Environment
python -m venv venv

# Activate Virtual Environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the configuration template:
```bash
cp .env.example .env
```
Key `.env` options:
- `DEMO_MODE=true`: Runs interactive presentation mode (works without full PyTorch model weights).
- `DEMO_MODE=false`: Strict ML execution mode (requires model weights loaded).
- `AASIST_CHECKPOINT_PATH=model_weights/aasist.pth`
- `SPECTRAL_MODEL_PATH=model_weights/spectral_clf.joblib`

---

## 💻 Running the Application

### Start Backend FastAPI Server
```bash
python -m backend.main
# Or via uvicorn directly:
# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
```
http://localhost:8000
```

---

## 🧪 Testing, Training & System Evaluation

### 1. Generate Sample Test Dataset
Populate `datasets/real/` and `datasets/fake/` with synthetic test audio files:
```bash
python datasets/generate_samples.py
```

### 2. Train Layer 3 Spectral LFCC Classifier
Train the Scikit-Learn classifier on the dataset:
```bash
python train_spectral.py --dataset_dir datasets/
```
Output model file saved to: `model_weights/spectral_clf.joblib`.

### 3. Evaluate System Performance
Calculate Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, and Confusion Matrix comparing individual layers against the 3-Layer Ensemble:
```bash
python evaluate.py --dataset_dir datasets/
```

### 4. Run Automated Test Suite
```bash
pytest tests/
```

---

## 📡 API Reference

### 1. Analyze Audio
`POST /api/analyze`

**Request Payload**: Multipart Form Data (`file`: Audio file binary).

**Example Response**:
```json
{
  "prediction": "Likely AI-Generated",
  "ai_probability": 0.87,
  "confidence": 0.87,
  "status": "success",
  "demo_mode": false,
  "layers": {
    "acoustic": 0.72,
    "waveform": 0.91,
    "spectral": 0.84
  },
  "layer_details": { ... },
  "audio": {
    "duration": 12.4,
    "format": "wav",
    "sample_rate": 16000,
    "channels": 1
  },
  "explanation": "The ensemble system analyzed the audio across 3 detection layers...",
  "processing_time_sec": 0.34,
  "anomalies": [
    "Unnaturally rigid pitch contour with near-zero micro-vibrato",
    "Abrupt high-frequency attenuation above 7.5 kHz"
  ]
}
```

### 2. Health Check
`GET /api/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "demo_mode": true,
  "models": {
    "layer1_acoustic": { "configured": true },
    "layer2_waveform_aasist": { "configured": false },
    "layer3_spectral_lfcc": { "configured": true }
  }
}
```

---

## 🔒 Security & Data Privacy Policy

1. **Volatile Processing**: Uploaded audio is saved to temporary operating system scratch directories inside isolated context managers and **immediately deleted after analysis**.
2. **No Data Retention**: Voice recordings are never stored permanently unless explicitly configured.
3. **Information Disclosure Protection**: Server filesystem paths, secret environment keys, and internal tracebacks are never exposed to API responses.

---

## 📜 License & Citation

VoiceShield is developed as an open-source research prototype for AI audio security and deepfake detection auditing.
