from typing import Any, Optional
from pydantic import BaseModel, Field
from app.schemas.evidence import Confidence, EvidenceItem


class WorkflowStep(BaseModel):
    step_number: int
    tool: str
    description: str
    status: str = "pending"  # pending, running, success, failure
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class WorkflowPlan(BaseModel):
    task: str
    steps: list[WorkflowStep]
    estimated_duration_ms: Optional[float] = None


class TraceEvent(BaseModel):
    step: int
    event: str
    status: str = "success"
    task: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[list[str]] = None
    duration_ms: Optional[float] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    job_id: str
    trace: list[TraceEvent] = Field(default_factory=list)


class ExecutionSummary(BaseModel):
    task: str
    models: list[str]
    tools: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0


class AnalysisResult(BaseModel):
    job_id: str
    answer: str
    task: str
    observations: list[dict[str, Any]] = Field(default_factory=list)
    inferences: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: Confidence
    evidence: list[EvidenceItem] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    execution_summary: ExecutionSummary
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    code: str
    message: str
    recoverable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
