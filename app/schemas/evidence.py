from typing import Any, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str
    type: str  # visual_region, change_map, segmentation_mask, metadata_field, cross_model_signal
    description: str
    claim: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    artifact_id: Optional[str] = None
    bbox: Optional[list[float]] = None  # [ymin, xmin, ymax, xmax] or [min_x, min_y, max_x, max_y]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConfidenceFactors(BaseModel):
    model_confidence: float = Field(0.5, ge=0.0, le=1.0)
    evidence_strength: float = Field(0.5, ge=0.0, le=1.0)
    input_quality: float = Field(0.5, ge=0.0, le=1.0)
    cross_model_agreement: float = Field(0.5, ge=0.0, le=1.0)
    task_suitability: float = Field(0.5, ge=0.0, le=1.0)


class Confidence(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: str = Field(..., description="high, medium, low, uncertain")
    label: str = Field("estimated confidence", description="Must be 'estimated confidence' unless calibrated")
    factors: ConfidenceFactors
    reasoning: str = ""
