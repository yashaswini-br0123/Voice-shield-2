# AGENTS.md - VoiceShield AI Deepfake Detection Governance & Rules

## Project Architecture & Overview
VoiceShield is an advanced 4-layer multimodal AI voice deepfake detection & prevention platform.
Pipeline: Audio Preprocessing → Parallel Feature Extraction (Acoustic, LFCC, WavLM SSL, AASIST) → Score Calibration → Ensemble Fusion → Forensic Report & Final Assessment.

---

## AI Detection Core Rules & Model Semantics

### 1. Model Input & Preprocessing Specifications
- **Common Standardization**: All incoming audio (WAV, MP3, AAC, M4A, OPUS, mic recordings) is converted to 16 kHz mono float32 array and peak-normalized.
- **Model-Specific Preprocessing**:
  - **Acoustic & LFCC**: Active speech frames (`|audio| > 0.02`) are isolated to trim room silence before calculating pitch, jitter, and cepstral variance.
  - **AASIST**: Raw waveform input is processed according to AASIST architectural input requirements (16 kHz mono, padded/truncated to 48,000 samples / 3.0s).

### 2. Model Label & Logit Semantics
- **AASIST (Anti-Spoofing Raw Waveform Network)**:
  - **Verified Repository Convention**: `logits[0]` = Spoof (AI), `logits[1]` = Bona-fide (Human).
  - **AI Risk Calculation**: `P(AI) = softmax(logits)[0]` (or `1 - softmax(logits)[1]`). Never invert this mapping.

### 3. WavLM SSL Representation & Downstream Classifier Rule
- **Self-Supervised Feature Extractor**: `torchaudio.pipelines.WAVLM_BASE` extracts 768-dimensional speech representations ($F_3$).
- **Strict Classification Head Rule**: WavLM embeddings MUST ONLY produce an AI detection score if connected to a genuinely trained downstream deepfake classification head (e.g. `wavlm_classifier.pth`).
- **Unconfigured Fallback**: If no trained WavLM classifier checkpoint exists, WavLM operates as representation profiling (`status: "representation_profiling"`, `score: None`). It must NOT be assigned an arbitrary weight or fake score in ensemble fusion.

### 4. Ensemble Fusion, Nomenclature & Thresholds
- **Score Nomenclature**: The final score is referred to as **"AI Risk Score"** (0.0 to 1.0) rather than a calibrated probability unless explicit empirical calibration curves are applied.
- **Configurable Prototype Thresholds**:
  - `HUMAN_THRESHOLD = 0.35`
  - `AI_THRESHOLD = 0.65`
  - `DISAGREEMENT_THRESHOLD = 0.28` (standard deviation across active score layers)
- **Multi-Level Classification Outcomes**:
  - 🟢 **Likely Human**: AI Risk Score < 0.35 with low layer variance.
  - 🟡 **Uncertain / Verify**: AI Risk Score 0.35–0.65 OR high layer disagreement (`std(scores) > 0.28`).
  - 🔴 **Likely AI-Generated**: AI Risk Score > 0.65 with high synthetic confidence.

### 5. Forensic Analysis Report Integrity
- Every report MUST contain: Report ID, UTC Timestamp, AI Risk Score, 4-Layer Evidence Breakdown, Evidence Consistency Rating, Final Assessment, and Recommended Action.
- Never hardcode or mock random scores in configured mode. All metrics must originate from empirical calculations.
