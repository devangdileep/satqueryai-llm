from typing import Any
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.models.base import RemoteSensingModel
from app.schemas.models import ModelCapability
from app.schemas.tools import ToolResult, ToolResultArtifact


class ChangeChatAdapter(RemoteSensingModel):
    """ChangeChat Adapter.
    Supports bi-temporal change detection, change VQA, change captioning, and change localization.
    """

    @property
    def name(self) -> str:
        return "ChangeChat"

    @property
    def capabilities(self) -> ModelCapability:
        return ModelCapability(
            name="ChangeChat",
            type="change_analysis",
            tasks=[
                "change_detection",
                "change_vqa",
                "change_captioning",
                "change_localization",
                "multitemporal_change_vqa",
                "multitemporal_change_analysis",
            ],
            supported_modalities=["optical", "multispectral"],
            requires_images=2,
            description="Bitemporal remote sensing change analysis, change captioning, and change localization."
        )

    async def predict(
        self,
        inputs: dict[str, Any],
        task: str,
        parameters: dict[str, Any] | None = None
    ) -> ToolResult:
        query = inputs.get("query", "What changed between these images?")
        image_a = inputs.get("image_a", "")
        image_b = inputs.get("image_b", "")

        if settings.MODEL_BACKEND == "http":
            return await self._predict_http(image_a, image_b, query, task, parameters)
        else:
            return self._predict_mock(query, task)

    async def _predict_http(self, image_a: str, image_b: str, query: str, task: str, parameters: dict[str, Any] | None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=settings.MODEL_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    settings.CHANGECHAT_ENDPOINT,
                    json={"image_a": image_a, "image_b": image_b, "query": query, "task": task, "parameters": parameters or {}}
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool="changechat",
                    status="success",
                    result=data.get("result", {}),
                    artifacts=[ToolResultArtifact(**a) for a in data.get("artifacts", [])],
                    metadata={"model": "ChangeChat", "endpoint": settings.CHANGECHAT_ENDPOINT}
                )
        except Exception as e:
            logger.error("changechat_http_failed", error=str(e))
            return ToolResult(
                tool="changechat",
                status="error",
                error=f"ChangeChat HTTP inference failed: {str(e)}",
                metadata={"model": "ChangeChat"}
            )

    def _predict_mock(self, query: str, task: str) -> ToolResult:
        return ToolResult(
            tool="changechat",
            status="success",
            result={
                "change_description": "Expansion of built-up settlement structures and clearing of surrounding vegetation observed between the pre-change and post-change observations.",
                "changed_area_percentage": 14.8,
                "change_detected": True,
                "change_regions": [
                    {
                        "id": "region_1",
                        "type": "built_up_expansion",
                        "confidence": 0.91,
                        "bbox": [100, 120, 340, 420]
                    },
                    {
                        "id": "region_2",
                        "type": "vegetation_clearing",
                        "confidence": 0.88,
                        "bbox": [400, 450, 600, 650]
                    }
                ]
            },
            artifacts=[],
            metadata={"model": "ChangeChat", "mode": "mock"}
        )

    async def health_check(self) -> bool:
        if settings.MODEL_BACKEND == "mock":
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = settings.CHANGECHAT_ENDPOINT.replace("/predict", "/health")
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False
