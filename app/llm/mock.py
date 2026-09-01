from typing import Type, TypeVar
from pydantic import BaseModel
from app.llm.base import LLMProvider
from app.schemas.analysis import QueryAnalysis

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider returning deterministic structured Pydantic models for testing."""

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        temperature: float = 0.0
    ) -> T:
        prompt_lower = prompt.lower()

        if response_model == QueryAnalysis:
            if "change" in prompt_lower or "changed" in prompt_lower:
                return QueryAnalysis(
                    intent="change_analysis",
                    task="multitemporal_change_vqa",
                    target_entities=["settlement", "structures", "vegetation"],
                    requested_outputs=["change_description", "change_location"],
                    requires_spatial_evidence=True,
                    requires_temporal_reasoning=True,
                    requires_multimodal_reasoning=False,
                    raw_query=prompt
                )
            elif "where" in prompt_lower or "ground" in prompt_lower or "building" in prompt_lower or "water" in prompt_lower:
                return QueryAnalysis(
                    intent="object_localization",
                    task="region_grounding",
                    target_entities=["water", "building"],
                    requested_outputs=["grounding_boxes", "location"],
                    requires_spatial_evidence=True,
                    requires_temporal_reasoning=False,
                    requires_multimodal_reasoning=False,
                    raw_query=prompt
                )
            elif "sar" in prompt_lower or "optical" in prompt_lower or "both" in prompt_lower or "fuse" in prompt_lower:
                return QueryAnalysis(
                    intent="multimodal_fusion",
                    task="optical_sar_analysis",
                    target_entities=["built_up", "water"],
                    requested_outputs=["joint_classification", "fused_evidence"],
                    requires_spatial_evidence=True,
                    requires_temporal_reasoning=False,
                    requires_multimodal_reasoning=True,
                    raw_query=prompt
                )
            else:
                return QueryAnalysis(
                    intent="scene_understanding",
                    task="single_image_vqa",
                    target_entities=["land_cover", "objects"],
                    requested_outputs=["description"],
                    requires_spatial_evidence=False,
                    requires_temporal_reasoning=False,
                    requires_multimodal_reasoning=False,
                    raw_query=prompt
                )

        # Generic fallback using schema defaults
        return response_model.model_construct()
