import time
import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.config import settings
from backend.utils.security import validate_uploaded_file, save_temp_file, get_media_type
from backend.audio.preprocessing import load_and_preprocess_audio
from backend.models.acoustic_detector import AcousticDetector
from backend.models.aasist_detector import AASISTDetector
from backend.models.spectral_detector import SpectralDetector
from backend.models.image_detector import ImageDetector
from backend.models.video_detector import VideoDetector
from backend.ensemble.fusion import EnsembleFusion
from backend.api.models_schema import DetectionResponseSchema, HealthResponseSchema

router = APIRouter(prefix="/api", tags=["VoiceShield API"])

# Lazy singletons for fast serverless cold-start
_acoustic_model = None
_aasist_model = None
_spectral_model = None
_image_model = None
_video_model = None
_fusion_engine = None

def get_acoustic_model():
    global _acoustic_model
    if _acoustic_model is None:
        _acoustic_model = AcousticDetector()
    return _acoustic_model

def get_aasist_model():
    global _aasist_model
    if _aasist_model is None:
        _aasist_model = AASISTDetector()
    return _aasist_model

def get_spectral_model():
    global _spectral_model
    if _spectral_model is None:
        _spectral_model = SpectralDetector()
    return _spectral_model

def get_image_model():
    global _image_model
    if _image_model is None:
        _image_model = ImageDetector()
    return _image_model

def get_video_model():
    global _video_model
    if _video_model is None:
        _video_model = VideoDetector()
    return _video_model

def get_fusion_engine():
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = EnsembleFusion()
    return _fusion_engine


@router.get("/health", response_model=HealthResponseSchema)
@router.get("/api/health", response_model=HealthResponseSchema)
async def health_check():
    """
    Checks status of backend, multimodal detectors, and DEMO_MODE setting.
    """
    aasist = get_aasist_model()
    spectral = get_spectral_model()
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "demo_mode": settings.DEMO_MODE,
        "models": {
            "layer1_acoustic": {
                "configured": True,
                "gemini_api_configured": bool(settings.GEMINI_API_KEY)
            },
            "layer2_waveform_aasist": {
                "configured": aasist.is_loaded,
                "checkpoint_path": settings.AASIST_CHECKPOINT_PATH
            },
            "layer3_spectral_lfcc": {
                "configured": spectral.is_loaded,
                "model_path": settings.SPECTRAL_MODEL_PATH
            },
            "image_detector_fft_ela": {
                "configured": True
            },
            "video_multimodal_detector": {
                "configured": True
            }
        }
    }


@router.post("/analyze", response_model=DetectionResponseSchema)
@router.post("/api/analyze", response_model=DetectionResponseSchema)
async def analyze_media(

    file: UploadFile = File(...),
    layer1_weight: Optional[float] = Form(default=None),
    layer2_weight: Optional[float] = Form(default=None),
    layer3_weight: Optional[float] = Form(default=None)
):
    """
    Multimodal API endpoint for binary (HUMAN vs AI) classification.
    """
    start_time = time.time()
    
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read upload payload: {str(e)}"
        )

    # Validate file size & extension, detect media category
    media_type = validate_uploaded_file(file, contents)

    with save_temp_file(contents, file.filename or "upload.bin") as temp_path:
        
        if media_type == "image":
            # Image Deepfake Analysis
            score, img_details = get_image_model().analyze(temp_path)
            proc_time = round(time.time() - start_time, 3)

            # Binary outcome: > 0.50 AI-Generated, <= 0.50 Human / Authentic Real
            prediction = "Likely AI-Generated" if score > 0.50 else "Likely Human"
            confidence = round(0.50 + abs(score - 0.50) * 0.96, 4)

            anomalies = img_details.get("anomalies", [])
            verdict_label = "LIKELY AI-GENERATED" if score > 0.50 else "LIKELY HUMAN"
            explanation = (
                f"The system performed spatial, 2D FFT spectral frequency, and Error Level Analysis (ELA) "
                f"on the uploaded image ({img_details.get('dimensions', 'N/A')}) and classified it as {verdict_label} (Synthetic Probability: {int(score * 100)}%)."
            )

            return {
                "prediction": prediction,
                "ai_probability": score,
                "confidence": confidence,
                "media_type": "image",
                "status": "success",
                "demo_mode": settings.DEMO_MODE,
                "layers": {
                    "fft_spectral": img_details.get("fft_spectral_score"),
                    "ela_compression": img_details.get("ela_compression_score"),
                    "noise_covariance": img_details.get("noise_covariance_score")
                },
                "layer_details": {"image": img_details},
                "audio": None,
                "explanation": explanation,
                "processing_time_sec": proc_time,
                "anomalies": anomalies
            }

        elif media_type == "video":
            # Video Deepfake Multimodal Analysis
            score, vid_details = get_video_model().analyze(temp_path)
            proc_time = round(time.time() - start_time, 3)

            # Binary outcome: > 0.50 AI Deepfake Video, <= 0.50 Authentic Real Video
            prediction = "Likely AI-Generated" if score > 0.50 else "Likely Human"
            confidence = round(0.50 + abs(score - 0.50) * 0.96, 4)

            anomalies = vid_details.get("anomalies", [])
            has_aud = vid_details.get("has_audio_track", False)
            verdict_label = "LIKELY AI-GENERATED" if score > 0.50 else "LIKELY HUMAN"
            explanation = (
                f"The system evaluated {vid_details.get('frames_analyzed', 0)} keyframes and "
                f"fused multimodal features, classifying the video as {verdict_label} (Synthetic Probability: {int(score * 100)}%)."
            )

            return {
                "prediction": prediction,
                "ai_probability": score,
                "confidence": confidence,
                "media_type": "video",
                "status": "success",
                "demo_mode": settings.DEMO_MODE,
                "layers": {
                    "visual_frames": vid_details.get("visual_ai_prob"),
                    "audio_track": vid_details.get("audio_ai_prob")
                },
                "layer_details": {"video": vid_details},
                "audio": {"duration": vid_details.get("duration_sec", 0.0), "format": "mp4", "sample_rate": 16000, "channels": 1},
                "explanation": explanation,
                "processing_time_sec": proc_time,
                "anomalies": anomalies
            }

        else:
            # Standard Audio Deepfake Analysis
            audio_array, audio_meta = load_and_preprocess_audio(temp_path)
            
            s1, l1_details = get_acoustic_model().analyze(audio_array, sr=audio_meta["sample_rate"])
            s2, l2_details = get_aasist_model().analyze(audio_array, sr=audio_meta["sample_rate"])
            s3, l3_details = get_spectral_model().analyze(audio_array, sr=audio_meta["sample_rate"])
            
            engine = get_fusion_engine()
            if any(w is not None for w in [layer1_weight, layer2_weight, layer3_weight]):
                engine = EnsembleFusion(w1=layer1_weight, w2=layer2_weight, w3=layer3_weight)
                
            ensemble_result = engine.fuse_scores(l1_details, l2_details, l3_details)
            proc_time = round(time.time() - start_time, 3)

            return {
                "prediction": ensemble_result["prediction"],
                "ai_probability": ensemble_result["ai_probability"],
                "confidence": ensemble_result["confidence"],
                "media_type": "audio",
                "status": "success",
                "demo_mode": settings.DEMO_MODE,
                "layers": {
                    "acoustic": l1_details.get("score"),
                    "waveform": l2_details.get("score"),
                    "spectral": l3_details.get("score")
                },
                "layer_details": {
                    "acoustic": l1_details,
                    "waveform": l2_details,
                    "spectral": l3_details
                },
                "audio": audio_meta,
                "explanation": ensemble_result["explanation"],
                "processing_time_sec": proc_time,
                "anomalies": ensemble_result["all_anomalies"]
            }
