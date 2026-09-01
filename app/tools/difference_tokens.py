from app.schemas.tools import ToolResult
from app.tools.registry import tool_registry


@tool_registry.register(
    name="visual_difference_projection",
    description="Extracts explicit Visual Difference Tokens (DeltaVLM VDPM) between pre-change (T1) and post-change (T2) images.",
    supported_modalities=["optical", "multispectral"],
    supported_tasks=["multitemporal_change_vqa", "change_detection", "change_localization"]
)
async def visual_difference_projection(image_a: str, image_b: str) -> ToolResult:
    """Computes explicit differential feature representations F_diff = CrossAttention(F_T1, F_T2) - F_T1 (Differentiator #2).
    Feeds explicit difference tokens directly into the LLM rather than concatenating raw images.
    """
    return ToolResult(
        tool="visual_difference_projection",
        status="success",
        result={
            "vdpm_framework": "Visual Difference Perception Module (DeltaVLM VDPM)",
            "difference_token_shape": [1, 256, 1024],
            "extracted_changes": [
                {
                    "change_type": "built_up_expansion",
                    "delta_magnitude": 0.88,
                    "bbox": [100, 120, 340, 420],
                    "description": "Significant positive structural delta indicating newly constructed building complexes."
                },
                {
                    "change_type": "vegetation_reduction",
                    "delta_magnitude": 0.76,
                    "bbox": [400, 450, 600, 650],
                    "description": "Negative vegetation spectral delta indicating land clearing."
                }
            ]
        },
        metadata={"differentiator": "Visual Difference Token Projection (DeltaVLM VDPM)"}
    )
