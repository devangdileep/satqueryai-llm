import os
import uuid
from pathlib import Path
from PIL import Image, ImageDraw
from app.core.config import settings
from app.schemas.tools import ToolResult, ToolResultArtifact
from app.tools.registry import tool_registry


@tool_registry.register(
    name="region_grounding",
    description="Draws bounding boxes and visual highlights on imagery to ground observations."
)
async def region_grounding(image_path: str, boxes: list[list[float]], labels: list[str] | None = None) -> ToolResult:
    if not Path(image_path).exists():
        return ToolResult(
            tool="region_grounding",
            status="error",
            error=f"Image path not found: {image_path}"
        )

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            formatted_boxes = []
            for i, box in enumerate(boxes):
                if len(box) == 4:
                    # Check if normalized [ymin, xmin, ymax, xmax] in [0, 1000] scale (GeoChat format)
                    if max(box) <= 1000 and any(c > 1.0 for c in box):
                        ymin, xmin, ymax, xmax = box
                        y1, x1 = (ymin / 1000.0) * h, (xmin / 1000.0) * w
                        y2, x2 = (ymax / 1000.0) * h, (xmax / 1000.0) * w
                    else:
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    label = labels[i] if labels and i < len(labels) else f"Target #{i+1}"
                    draw.text((x1 + 4, y1 + 4), label, fill="yellow")

                    formatted_boxes.append({
                        "id": f"box_{i+1}",
                        "bbox": [x1, y1, x2, y2],
                        "label": label
                    })

            # Save annotated output artifact
            artifact_id = f"grounding_{uuid.uuid4().hex[:8]}.png"
            output_dir = Path(settings.STORAGE_PATH) / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / artifact_id
            img.save(output_path)

            return ToolResult(
                tool="region_grounding",
                status="success",
                result={
                    "box_count": len(formatted_boxes),
                    "grounded_regions": formatted_boxes
                },
                artifacts=[
                    ToolResultArtifact(
                        artifact_id=artifact_id,
                        type="image_overlay",
                        path=str(output_path),
                        description="Visual grounding overlay with bounding boxes."
                    )
                ]
            )
    except Exception as e:
        return ToolResult(
            tool="region_grounding",
            status="error",
            error=f"Region grounding rendering failed: {str(e)}"
        )
