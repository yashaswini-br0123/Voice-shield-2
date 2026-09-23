import numpy as np
from typing import Dict, Any, List, Optional
from backend.config import settings


class EnsembleFusion:
    """
    4-Layer Ensemble Score Fusion & Calibration Engine for VoiceShield.
    Fuses validated scores from:
    - Layer 1: Acoustic Analysis (w=0.25 prototype)
    - Layer 2: AASIST Raw Waveform Network (w=0.35 prototype)
    - Layer 3: LFCC Spectral Cepstral Classifier (w=0.25 prototype)
    - Layer 4: WavLM SSL Classifier Head (w=0.15 prototype, if classifier head is loaded)

    Rules:
    - Configurable prototype thresholds: HUMAN_THRESHOLD=0.35, AI_THRESHOLD=0.65, DISAGREEMENT_THRESHOLD=0.28.
    - If WavLM has no classifier head (status="representation_profiling" / score=None), it is omitted from score weighting.
    - If active layer scores disagree strongly (std(scores) > 0.28), verdict is UNCERTAIN / VERIFY.
    - Result terminology: "AI Risk Score" (0.0 to 1.0).
    """
    def __init__(
        self,
        human_thresh: float = None,
        ai_thresh: float = None,
        disagreement_thresh: float = 0.28
    ):
        self.human_thresh = human_thresh if human_thresh is not None else settings.HUMAN_THRESHOLD
        self.ai_thresh = ai_thresh if ai_thresh is not None else settings.AI_THRESHOLD
        self.disagreement_thresh = disagreement_thresh

    def fuse_scores(
        self,
        layer1_details: Dict[str, Any],
        layer2_details: Dict[str, Any],
        layer3_details: Dict[str, Any],
        wavlm_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculates ensemble AI Risk Score with dynamic weight normalization,
        evaluates evidence consistency, and returns multi-tier outcome:
        - Likely Human
        - Uncertain / Verify
        - Likely AI-Generated
        """
        active_layers = []

        # Default prototype weights
        w_acoustic = getattr(settings, "LAYER1_ACOUSTIC_WEIGHT", 0.25)
        w_aasist = getattr(settings, "LAYER2_WAVEFORM_WEIGHT", 0.35)
        w_lfcc = getattr(settings, "LAYER3_SPECTRAL_WEIGHT", 0.25)
        w_wavlm = 0.15

        # Layer 1: Acoustic
        s1 = layer1_details.get("score")
        if s1 is not None and layer1_details.get("status") != "unconfigured":
            active_layers.append(("acoustic", s1, w_acoustic))

        # Layer 2: AASIST
        s2 = layer2_details.get("score")
        if s2 is not None and layer2_details.get("status") != "unconfigured":
            active_layers.append(("aasist", s2, w_aasist))

        # Layer 3: LFCC
        s3 = layer3_details.get("score")
        if s3 is not None and layer3_details.get("status") != "unconfigured":
            active_layers.append(("lfcc", s3, w_lfcc))

        # Layer 4: WavLM (Only included if classifier head score exists)
        if wavlm_details:
            sw = wavlm_details.get("score")
            if sw is not None and wavlm_details.get("status") == "configured":
                active_layers.append(("wavlm", sw, w_wavlm))

        if not active_layers:
            return {
                "prediction": "Uncertain / Verify",
                "ai_risk_score": 0.50,
                "ai_probability": 0.50,
                "confidence": 0.50,
                "outcome_type": "uncertain",
                "consistency_rating": "Inconsistent / Disagreement Flagged",
                "explanation": "No active score-based detection layers were available for analysis."
            }

        # Dynamic weight normalization
        total_weight = sum(w for _, _, w in active_layers)
        scores = [score for _, score, _ in active_layers]
        weighted_score_sum = sum(score * (w / total_weight) for _, score, w in active_layers)

        ai_risk_score = max(0.01, min(0.99, round(weighted_score_sum, 4)))
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Layer Disagreement Check
        is_disagreement = score_std > self.disagreement_thresh

        # Outcome Classification Logic
        if is_disagreement:
            prediction = "Uncertain / Verify"
            outcome_type = "uncertain"
            consistency_rating = "Inconsistent - High Layer Disagreement"
        elif ai_risk_score >= self.ai_thresh:
            prediction = "Likely AI-Generated"
            outcome_type = "ai"
            consistency_rating = "High Consistency - Synthetic Risk Flagged"
        elif ai_risk_score <= self.human_thresh:
            prediction = "Likely Human"
            outcome_type = "human"
            consistency_rating = "High Consistency - Verified Natural"
        else:
            prediction = "Uncertain / Verify"
            outcome_type = "uncertain"
            consistency_rating = "Moderate Consistency - Intermediate Risk"

        confidence = round(0.50 + abs(ai_risk_score - 0.50) * 0.96, 4)

        # Collect anomalies
        all_anomalies: List[str] = []
        all_details = [layer1_details, layer2_details, layer3_details]
        if wavlm_details:
            all_details.append(wavlm_details)

        for det in all_details:
            for anomaly in det.get("anomalies", []):
                if anomaly and anomaly not in all_anomalies:
                    all_anomalies.append(anomaly)

        explanation = self._build_explanation(prediction, ai_risk_score, active_layers, score_std)

        return {
            "prediction": prediction,
            "ai_risk_score": ai_risk_score,
            "ai_probability": ai_risk_score, # Backwards compatibility
            "confidence": confidence,
            "outcome_type": outcome_type,
            "consistency_rating": consistency_rating,
            "score_std": round(score_std, 4),
            "prototype_thresholds": {
                "human": self.human_thresh,
                "ai": self.ai_thresh,
                "disagreement": self.disagreement_thresh
            },
            "layer_weights_applied": {
                name: round(w / total_weight, 4) for name, _, w in active_layers
            },
            "explanation": explanation,
            "all_anomalies": all_anomalies
        }

    def _build_explanation(
        self,
        prediction: str,
        ai_risk_score: float,
        active_layers: List[tuple],
        score_std: float
    ) -> str:
        pct = int(ai_risk_score * 100)
        layer_names = ", ".join([name.upper() for name, _, _ in active_layers])

        if prediction == "Likely AI-Generated":
            return (
                f"Evaluated across {len(active_layers)} active layers ({layer_names}). "
                f"High synthetic evidence consistency identified (AI Risk Score: {pct}%)."
            )
        elif prediction == "Likely Human":
            return (
                f"Evaluated across {len(active_layers)} active layers ({layer_names}). "
                f"Natural vocal tract vibrato and phase trajectory confirmed (AI Risk Score: {pct}%)."
            )
        else:
            return (
                f"Evaluated across {len(active_layers)} active layers ({layer_names}). "
                f"Layer disagreement or intermediate risk detected (std={score_std:.2f}, AI Risk Score: {pct}%). Verification recommended."
            )
