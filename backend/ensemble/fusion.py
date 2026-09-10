from typing import Dict, Any, List
from backend.config import settings


class EnsembleFusion:
    """
    Ensemble Score Fusion System for VoiceShield.
    Fuses probabilities from Layer 1 (Acoustic), Layer 2 (Waveform/Phase), and Layer 3 (Spectral).
    Binary Decision Boundary at 0.50 (HUMAN vs AI).
    """
    def __init__(
        self,
        w1: float = None,
        w2: float = None,
        w3: float = None,
        decision_thresh: float = 0.50
    ):
        self.w1 = w1 if w1 is not None else settings.LAYER1_ACOUSTIC_WEIGHT
        self.w2 = w2 if w2 is not None else settings.LAYER2_WAVEFORM_WEIGHT
        self.w3 = w3 if w3 is not None else settings.LAYER3_SPECTRAL_WEIGHT
        self.decision_thresh = decision_thresh

    def fuse_scores(
        self,
        layer1_details: Dict[str, Any],
        layer2_details: Dict[str, Any],
        layer3_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates ensemble AI probability with dynamic weight normalization,
        classifies binary prediction (HUMAN vs AI), and constructs technical explanation.
        """
        active_layers = []
        
        # Check active status of Layer 1
        s1 = layer1_details.get("score")
        if s1 is not None and layer1_details.get("status") != "unconfigured":
            active_layers.append(("acoustic", s1, self.w1))

        # Check active status of Layer 2
        s2 = layer2_details.get("score")
        if s2 is not None and layer2_details.get("status") != "unconfigured":
            active_layers.append(("waveform", s2, self.w2))

        # Check active status of Layer 3
        s3 = layer3_details.get("score")
        if s3 is not None and layer3_details.get("status") != "unconfigured":
            active_layers.append(("spectral", s3, self.w3))

        if not active_layers:
            return {
                "prediction": "Likely Human",
                "ai_probability": 0.5,
                "confidence": 0.5,
                "status": "error",
                "message": "No detection layers were available for analysis."
            }

        # Normalize weights across active layers
        total_weight = sum(w for _, _, w in active_layers)
        weighted_score_sum = sum(score * (w / total_weight) for _, score, w in active_layers)
        
        final_ai_prob = max(0.01, min(0.99, round(weighted_score_sum, 4)))

        # Direct Binary Classification (HUMAN vs AI)
        if final_ai_prob > self.decision_thresh:
            prediction = "Likely AI-Generated"
            confidence = round(0.50 + (final_ai_prob - 0.50) * 0.96, 4)
        else:
            prediction = "Likely Human"
            confidence = round(0.50 + (0.50 - final_ai_prob) * 0.96, 4)

        # Collect all technical anomaly findings
        all_anomalies: List[str] = []
        for det in [layer1_details, layer2_details, layer3_details]:
            for anomaly in det.get("anomalies", []):
                if anomaly and anomaly not in all_anomalies:
                    all_anomalies.append(anomaly)

        # Formulate human-readable explanation
        explanation = self._build_explanation(prediction, final_ai_prob, active_layers, all_anomalies)

        return {
            "prediction": prediction,
            "ai_probability": final_ai_prob,
            "confidence": confidence,
            "decision_threshold": self.decision_thresh,
            "layer_weights_applied": {
                name: round(w / total_weight, 4) for name, _, w in active_layers
            },
            "explanation": explanation,
            "all_anomalies": all_anomalies
        }

    def _build_explanation(
        self,
        prediction: str,
        ai_prob: float,
        active_layers: List[tuple],
        anomalies: List[str]
    ) -> str:
        pct = int(ai_prob * 100)
        layer_names = ", ".join([name.capitalize() for name, _, _ in active_layers])

        if prediction == "Likely AI-Generated":
            return (
                f"The ensemble system analyzed the sample across {len(active_layers)} detection layers ({layer_names}) "
                f"and classified it as LIKELY AI-GENERATED (Synthetic Probability: {pct}%)."
            )
        else:
            return (
                f"The ensemble system evaluated the sample across {len(active_layers)} detection layers ({layer_names}) "
                f"and classified it as LIKELY HUMAN (Synthetic Probability: {pct}%)."
            )
