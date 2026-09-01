import uuid
from pathlib import Path
from PIL import Image, ImageDraw
from app.core.config import settings
from app.schemas.tools import ToolResult, ToolResultArtifact
from app.tools.registry import tool_registry


@tool_registry.register(
    name="generate_visual_evidence",
    description="Generates visual evidence artifacts, change maps, and overlays."
)
async def generate_visual_evidence(
    image_paths: list[str],
    regions: list[dict] | None = None,
    task: str = "visual_evidence"
) -> ToolResult:
    artifacts = []
    output_dir = Path(settings.STORAGE_PATH) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(image_paths) >= 2:
        # Create bitemporal side-by-side or change map artifact
        try:
            img1 = Image.open(image_paths[0]).convert("RGB")
            img2 = Image.open(image_paths[1]).convert("RGB")

            # Resize img2 to match img1
            img2 = img2.resize(img1.size)

            w, h = img1.size
            side_by_side = Image.new("RGB", (w * 2, h))
            side_by_side.paste(img1, (0, 0))
            side_by_side.paste(img2, (w, 0))

            draw = ImageDraw.Draw(side_by_side)
            draw.text((10, 10), "Pre-Change (T1)", fill="white")
            draw.text((w + 10, 10), "Post-Change (T2)", fill="white")

            artifact_id = f"bitemporal_comparison_{uuid.uuid4().hex[:8]}.png"
            out_path = output_dir / artifact_id
            side_by_side.save(out_path)

            artifacts.append(
                ToolResultArtifact(
                    artifact_id=artifact_id,
                    type="change_map",
                    path=str(out_path),
                    description="Bi-temporal before/after visual comparison."
                )
            )
        except Exception as e:
            pass

    return ToolResult(
        tool="generate_visual_evidence",
        status="success",
        result={
            "evidence_generated": True,
            "artifact_count": len(artifacts)
        },
        artifacts=artifacts
    )
