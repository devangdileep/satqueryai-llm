from typing import Optional
from pydantic import BaseModel, Field


class ModelCapability(BaseModel):
    name: str
    type: str  # vision_language, change_analysis, earth_observation_foundation_model, multimodal_fusion, geospatial_toolkit
    tasks: list[str]
    supported_modalities: list[str]
    requires_images: int = 1
    requires_modalities: Optional[list[str]] = None
    description: str = ""


class ModelSelection(BaseModel):
    selected_model: str
    task: str
    reasoning: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    fallback_models: list[str] = Field(default_factory=list)


class ModelStatus(BaseModel):
    name: str
    status: str  # available, unavailable, mock
    endpoint: Optional[str] = None
    capabilities: ModelCapability
    healthy: bool = True
    reason: Optional[str] = None
