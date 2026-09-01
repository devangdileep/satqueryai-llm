from typing import Any
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.models.base import RemoteSensingModel
from app.schemas.models import ModelCapability
from app.schemas.tools import ToolResult, ToolResultArtifact


class GeoChatAdapter(RemoteSensingModel):
    """GeoChat Adapter.
    Supports single-image VQA, remote-sensing captioning, and region grounding.
    Can operate via HTTP endpoint or Mock mode.
    """

    @property
    def name(self) -> str:
        return "GeoChat"

    @property
    def capabilities(self) -> ModelCapability:
        return ModelCapability(
            name="GeoChat",
            type="vision_language",
            tasks=[
                "single_image_vqa",
                "captioning",
                "remote_sensing_understanding",
                "region_grounding",
                "object_localization",
                "scene_understanding",
            ],
            supported_modalities=["optical", "multispectral", "sar"],
            requires_images=1,
            description="Single-image VQA, captioning, scene understanding, and region grounding."
        )

    async def predict(
        self,
        inputs: dict[str, Any],
        task: str,
        parameters: dict[str, Any] | None = None
    ) -> ToolResult:
        query = inputs.get("query", "Describe this image.")
        image_path = inputs.get("image_path", "")

        if settings.MODEL_BACKEND == "http":
            return await self._predict_http(image_path, query, task, parameters)
        else:
            return self._predict_mock(query, task)

    async def _predict_http(self, image_path: str, query: str, task: str, parameters: dict[str, Any] | None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=settings.MODEL_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    settings.GEOCHAT_ENDPOINT,
                    json={"image_path": image_path, "query": query, "task": task, "parameters": parameters or {}}
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool="geochat",
                    status="success",
                    result=data.get("result", {}),
                    artifacts=[ToolResultArtifact(**a) for a in data.get("artifacts", [])],
                    metadata={"model": "GeoChat", "endpoint": settings.GEOCHAT_ENDPOINT}
                )
        except Exception as e:
            logger.error("geochat_http_failed", error=str(e))
            return ToolResult(
                tool="geochat",
                status="error",
                error=f"GeoChat HTTP inference failed: {str(e)}",
                metadata={"model": "GeoChat"}
            )

    def _predict_mock(self, query: str, task: str) -> ToolResult:
        query_lower = query.lower()
        if "water" in query_lower:
            text = "The satellite imagery displays water bodies along with surrounding vegetation and infrastructure."
            boxes = [[120, 150, 450, 600]]
        elif "building" in query_lower or "structure" in query_lower or "where" in query_lower:
            text = "Multiple built-up structures and residential buildings are identified in the center of the scene."
            boxes = [[200, 220, 380, 410], [500, 520, 680, 710]]
        else:
            text = "High-resolution optical satellite image showcasing mixed urban settlement, agricultural patches, and transport networks."
            boxes = []

        return ToolResult(
            tool="geochat",
            status="success",
            result={
                "response_text": text,
                "grounded_boxes": boxes,
                "confidence": 0.92,
                "identified_categories": ["built_up", "vegetation", "roads"]
            },
            artifacts=[],
            metadata={"model": "GeoChat", "mode": "mock"}
        )

    async def health_check(self) -> bool:
        if settings.MODEL_BACKEND == "mock":
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = settings.GEOCHAT_ENDPOINT.replace("/predict", "/health")
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False
