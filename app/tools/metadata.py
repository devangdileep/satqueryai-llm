from pathlib import Path
from app.schemas.tools import ToolResult
from app.tools.registry import tool_registry


@tool_registry.register(
    name="classify_image_modality",
    description="Classifies image modality into optical, multispectral, SAR, or unknown."
)
async def classify_image_modality(file_path: str, user_metadata: dict | None = None) -> ToolResult:
    filename = Path(file_path).name.lower()
    modality = "unknown"
    confidence = 0.5
    source = "band_heuristic"

    # 1. Check user metadata if provided
    if user_metadata and "modality" in user_metadata:
        m = str(user_metadata["modality"]).lower()
        if m in ["optical", "multispectral", "sar"]:
            return ToolResult(
                tool="classify_image_modality",
                status="success",
                result={
                    "modality": m,
                    "confidence": 0.95,
                    "source": "metadata"
                }
            )

    # 2. Check filename heuristics
    if any(k in filename for k in ["sar", "sentinel1", "s1", "vv", "vh", "radar"]):
        modality = "sar"
        confidence = 0.90
        source = "filename_heuristic"
    elif any(k in filename for k in ["hls", "sentinel2", "s2", "landsat", "msi"]):
        modality = "multispectral"
        confidence = 0.88
        source = "filename_heuristic"
    elif any(k in filename for k in ["rgb", "optical", "photo", "png", "jpg", "jpeg"]):
        modality = "optical"
        confidence = 0.85
        source = "filename_heuristic"
    else:
        modality = "optical"
        confidence = 0.70
        source = "default_fallback"

    return ToolResult(
        tool="classify_image_modality",
        status="success",
        result={
            "modality": modality,
            "confidence": confidence,
            "source": source
        }
    )
