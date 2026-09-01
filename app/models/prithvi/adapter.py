from typing import Any
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.models.base import RemoteSensingModel
from app.schemas.models import ModelCapability
from app.schemas.tools import ToolResult, ToolResultArtifact


class PrithviAdapter(RemoteSensingModel):
    """Prithvi-EO-2.0 Adapter.
    Supports Earth Observation representation, feature extraction, multispectral segmentation.
    """

    @property
    def name(self) -> str:
        return "Prithvi"

    @property
    def capabilities(self) -> ModelCapability:
        return ModelCapability(
            name="Prithvi",
            type="earth_observation_foundation_model",
            tasks=[
                "feature_extraction",
                "segmentation",
                "multispectral_analysis",
                "land_cover_classification",
            ],
            supported_modalities=["optical", "multispectral"],
            requires_images=1,
            description="Foundation model for feature extraction, multispectral representations, and dense semantic segmentation."
        )

    async def predict(
        self,
        inputs: dict[str, Any],
        task: str,
        parameters: dict[str, Any] | None = None
    ) -> ToolResult:
        image_path = inputs.get("image_path", "")

        if settings.MODEL_BACKEND == "http":
            return await self._predict_http(image_path, task, parameters)
        else:
            return self._predict_mock(task)

    async def _predict_http(self, image_path: str, task: str, parameters: dict[str, Any] | None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=settings.MODEL_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    settings.PRITHVI_ENDPOINT,
                    json={"image_path": image_path, "task": task, "parameters": parameters or {}}
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool="prithvi",
                    status="success",
                    result=data.get("result", {}),
                    artifacts=[ToolResultArtifact(**a) for a in data.get("artifacts", [])],
                    metadata={"model": "Prithvi", "endpoint": settings.PRITHVI_ENDPOINT}
                )
        except Exception as e:
            logger.error("prithvi_http_failed", error=str(e))
            return ToolResult(
                tool="prithvi",
                status="error",
                error=f"Prithvi HTTP inference failed: {str(e)}",
                metadata={"model": "Prithvi"}
            )

    def _predict_mock(self, task: str) -> ToolResult:
        return ToolResult(
            tool="prithvi",
            status="success",
            result={
                "task": task,
                "feature_embedding_shape": [1, 1024, 14, 14],
                "detected_classes": [
                    {"class_id": 1, "name": "water", "coverage_percentage": 22.5},
                    {"class_id": 2, "name": "built_up", "coverage_percentage": 45.0},
                    {"class_id": 3, "name": "vegetation", "coverage_percentage": 32.5}
                ],
                "segmentation_confidence": 0.94
            },
            artifacts=[],
            metadata={"model": "Prithvi", "mode": "mock"}
        )

    async def health_check(self) -> bool:
        if settings.MODEL_BACKEND == "mock":
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = settings.PRITHVI_ENDPOINT.replace("/predict", "/health")
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False
