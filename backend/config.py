import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "VoiceShield - Multimodal AI Deepfake Detection Platform"
    VERSION: str = "2.0.0"
    
    # Operation Mode
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Base Directory
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Model Weights Paths
    AASIST_CHECKPOINT_PATH: str = os.getenv("AASIST_CHECKPOINT_PATH", os.path.join(BASE_DIR, "model_weights", "aasist.pth"))
    SPECTRAL_MODEL_PATH: str = os.getenv("SPECTRAL_MODEL_PATH", os.path.join(BASE_DIR, "model_weights", "spectral_clf.joblib"))
    
    # Optional Gemini multimodal API key
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Detection Thresholds
    HUMAN_THRESHOLD: float = float(os.getenv("HUMAN_THRESHOLD", "0.35"))
    AI_THRESHOLD: float = float(os.getenv("AI_THRESHOLD", "0.65"))
    
    # Ensemble Layer Weights for Audio
    LAYER1_ACOUSTIC_WEIGHT: float = float(os.getenv("LAYER1_ACOUSTIC_WEIGHT", "0.30"))
    LAYER2_WAVEFORM_WEIGHT: float = float(os.getenv("LAYER2_WAVEFORM_WEIGHT", "0.40"))
    LAYER3_SPECTRAL_WEIGHT: float = float(os.getenv("LAYER3_SPECTRAL_WEIGHT", "0.30"))
    
    # Universal Audio Extensions
    AUDIO_EXTENSIONS: List[str] = [
        "wav", "mp3", "m4a", "flac", "ogg", "webm", "aac",
        "wma", "opus", "amr", "aiff", "caf", "3gp", "m4b", "mp2", "au"
    ]
    IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    VIDEO_EXTENSIONS: List[str] = ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v"]
    
    # Combined Allowed Extensions
    ALLOWED_EXTENSIONS: List[str] = [
        "wav", "mp3", "m4a", "flac", "ogg", "webm", "aac", "wma", "opus", "amr", "aiff", "caf", "3gp", "m4b", "mp2", "au",
        "jpg", "jpeg", "png", "webp", "bmp", "tiff",
        "mp4", "avi", "mov", "mkv", "flv", "wmv", "m4v"
    ]
    
    # File Constraints
    MAX_FILE_SIZE_MB: float = float(os.getenv("MAX_FILE_SIZE_MB", "50.0"))
    MAX_AUDIO_DURATION_SEC: float = float(os.getenv("MAX_AUDIO_DURATION_SEC", "120.0"))
    MIN_AUDIO_DURATION_SEC: float = float(os.getenv("MIN_AUDIO_DURATION_SEC", "0.5"))
    
    # Audio target sample rate
    TARGET_SAMPLE_RATE: int = 16000
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }


settings = Settings()
