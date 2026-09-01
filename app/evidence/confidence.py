from app.schemas.evidence import Confidence, ConfidenceFactors


class ConfidenceEngine:
    """Dedicated confidence calculation engine.
    Considers model confidence, evidence strength, input quality, cross-model agreement, and task suitability.
    Score is explicitly labeled as 'estimated confidence' unless calibrated.
    """

    def calculate(
        self,
        model_confidence: float = 0.9,
        evidence_strength: float = 0.85,
        input_quality: float = 0.9,
        cross_model_agreement: float = 0.85,
        task_suitability: float = 0.95
    ) -> Confidence:
        factors = ConfidenceFactors(
            model_confidence=model_confidence,
            evidence_strength=evidence_strength,
            input_quality=input_quality,
            cross_model_agreement=cross_model_agreement,
            task_suitability=task_suitability
        )

        # Weighted calculation
        score = (
            model_confidence * 0.30 +
            evidence_strength * 0.25 +
            input_quality * 0.15 +
            cross_model_agreement * 0.15 +
            task_suitability * 0.15
        )
        score = round(min(max(score, 0.0), 1.0), 2)

        if score >= 0.85:
            level = "high"
            reasoning = "High model confidence backed by strong visual evidence and clean input quality."
        elif score >= 0.70:
            level = "medium"
            reasoning = "Moderate confidence; primary observations supported by evidence."
        elif score >= 0.50:
            level = "low"
            reasoning = "Low confidence due to weak visual signals or input quality limitations."
        else:
            level = "uncertain"
            reasoning = "Uncertain; available imagery evidence is insufficient for reliable interpretation."

        return Confidence(
            score=score,
            level=level,
            label="estimated confidence",
            factors=factors,
            reasoning=reasoning
        )


confidence_engine = ConfidenceEngine()
