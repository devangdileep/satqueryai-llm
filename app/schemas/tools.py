from typing import Any, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    supported_modalities: list[str] = Field(default_factory=list)
    supported_tasks: list[str] = Field(default_factory=list)


class ToolResultArtifact(BaseModel):
    artifact_id: str
    type: str  # image_overlay, change_map, mask, report, cropped_region
    path: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool: str
    status: str  # success, failure, error
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ToolResultArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
