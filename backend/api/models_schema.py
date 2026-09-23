from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class DetectionResponseSchema(BaseModel):
    prediction: str = Field(..., description="'Likely AI-Generated', 'Likely Human', or 'Uncertain'")
    ai_probability: float = Field(..., description="Overall AI probability [0.0 - 1.0]")
    ai_risk_score: Optional[float] = Field(None, description="Calibrated AI Risk Score [0.0 - 1.0]")
    confidence: float = Field(..., description="Confidence score [0.0 - 1.0]")
    outcome_type: Optional[str] = Field(None, description="Multilevel outcome category")
    consistency_rating: Optional[str] = Field(None, description="Cross-layer evidence consistency rating")
    media_type: str = Field("audio", description="Detected media category ('audio', 'image', 'video')")
    status: str = Field("success", description="Execution status")
    demo_mode: bool = Field(False, description="True if operating in demo presentation mode")
    layers: Dict[str, Any] = Field(default_factory=dict, description="Layer probability breakdowns")
    layer_details: Dict[str, Any] = Field(default_factory=dict, description="Technical layer details")
    audio: Optional[Dict[str, Any]] = Field(None, description="Audio metadata profile")
    explanation: str = Field(..., description="Human-readable explanation of verdict")
    processing_time_sec: float = Field(..., description="Processing duration in seconds")
    anomalies: List[str] = Field(default_factory=list, description="Detected spectro-temporal or visual anomalies")


class HealthResponseSchema(BaseModel):
    status: str
    version: str
    demo_mode: bool
    models: Dict[str, Any]
