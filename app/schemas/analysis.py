from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    intent: str = Field(
        ...,
        description="e.g. change_analysis, scene_understanding, object_localization, multimodal_fusion"
    )
    task: str = Field(
        ...,
        description="e.g. multitemporal_change_vqa, single_image_vqa, region_grounding, optical_sar_analysis"
    )
    target_entities: list[str] = Field(default_factory=list, description="e.g. ['settlement', 'water', 'building']")
    requested_outputs: list[str] = Field(
        default_factory=list,
        description="e.g. ['change_description', 'change_location', 'count']"
    )
    requires_spatial_evidence: bool = False
    requires_temporal_reasoning: bool = False
    requires_multimodal_reasoning: bool = False
    raw_query: str


class TaskSpecification(BaseModel):
    task_type: str
    query_analysis: QueryAnalysis
    input_config: Any  # ImageInputConfig
    required_modalities: list[str]
    required_image_count: int
    required_capabilities: list[str]
