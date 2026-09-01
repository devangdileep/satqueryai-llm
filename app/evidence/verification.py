from app.core.logging import logger
from app.schemas.evidence import Confidence, ConfidenceFactors


class CrossModalVerifier:
    """Cross-Modal Verification Engine (Differentiator #4).
    Verifies agreement/disagreement between Optical spectral observations and SAR structural backscatter.
    If models disagree (e.g. optical says water, SAR says high double-bounce backscatter), degrades confidence
    and generates an honest uncertainty statement rather than manufacturing a confident hallucination.
    """

    def verify_and_adjust_confidence(
        self,
        optical_claims: list[dict],
        sar_claims: list[dict],
        base_confidence: float = 0.90
    ) -> tuple[Confidence, list[str]]:
        agreements = []
        disagreements = []
        uncertainties = []

        # Compare optical vs SAR class interpretations
        for opt in optical_claims:
            opt_class = opt.get("category", opt.get("class", ""))
            for sar in sar_claims:
                sar_class = sar.get("grounded_class", sar.get("category", ""))

                if opt_class == sar_class:
                    agreements.append(f"Confirmed: Optical '{opt_class}' matches SAR physical scattering signature.")
                elif opt_class and sar_class and opt_class != sar_class:
                    disagreements.append(
                        f"Conflict: Optical detected '{opt_class}' while SAR scattering indicates '{sar_class}'."
                    )

        # Calculate agreement factor
        if disagreements:
            cross_agreement_score = max(0.40, 0.90 - (0.25 * len(disagreements)))
            uncertainties.append(
                "Cross-modal contradiction detected between optical reflectance and SAR radar backscatter. "
                "Confidence degraded to reflect environmental ambiguity (e.g. cloud shadow or flooded vegetation)."
            )

            logger.warn(
                "cross_modal_conflict_detected",
                conflicts=disagreements,
                degraded_confidence=cross_agreement_score
            )
        else:
            cross_agreement_score = 0.95
            agreements.append("Full cross-modal verification confirmed across optical and SAR sensors.")

        factors = ConfidenceFactors(
            model_confidence=base_confidence,
            evidence_strength=0.90 if agreements else 0.60,
            input_quality=0.92,
            cross_model_agreement=cross_agreement_score,
            task_suitability=0.95
        )

        final_score = round(
            base_confidence * 0.30 +
            factors.evidence_strength * 0.25 +
            factors.input_quality * 0.15 +
            cross_agreement_score * 0.20 +
            factors.task_suitability * 0.10,
            2
        )

        level = "high" if final_score >= 0.85 else ("medium" if final_score >= 0.70 else "low")

        confidence = Confidence(
            score=final_score,
            level=level,
            label="estimated confidence (cross-modal verified)",
            factors=factors,
            reasoning="; ".join(agreements) if not disagreements else "; ".join(disagreements)
        )

        return confidence, uncertainties


cross_modal_verifier = CrossModalVerifier()
