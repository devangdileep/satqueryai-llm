from typing import Any
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.models.base import RemoteSensingModel
from app.schemas.models import ModelCapability
from app.schemas.tools import ToolResult, ToolResultArtifact


class SARFusionAdapter(RemoteSensingModel):
    """SAR-ML-Fusion Adapter.
    Supports Optical + SAR multimodal analysis, joint land cover classification, cloud penetration, and built-up/water detection.
    """

    @property
    def name(self) -> str:
        return "SAR-ML-Fusion"

    @property
    def capabilities(self) -> ModelCapability:
        return ModelCapability(
            name="SAR-ML-Fusion",
            type="multimodal_fusion",
            tasks=[
                "optical_sar_analysis",
                "cross_modal_analysis",
                "multimodal_land_cover",
                "joint_reasoning",
            ],
            supported_modalities=["optical", "sar"],
            requires_images=2,
            requires_modalities=["optical", "sar"],
            description="Optical + SAR joint multimodal fusion for complementary structural and spectral analysis."
        )

    async def predict(
        self,
        inputs: dict[str, Any],
        task: str,
        parameters: dict[str, Any] | None = None
    ) -> ToolResult:
        optical_path = inputs.get("optical_path", "")
        sar_path = inputs.get("sar_path", "")

        if settings.MODEL_BACKEND == "http":
            return await self._predict_http(optical_path, sar_path, task, parameters)
        else:
            return self._predict_mock(task)

    async def _predict_http(self, optical_path: str, sar_path: str, task: str, parameters: dict[str, Any] | None) -> ToolResult:
        try:
            async with httpx.AsyncClient(timeout=settings.MODEL_HTTP_TIMEOUT) as client:
                resp = await client.post(
                    settings.SAR_FUSION_ENDPOINT,
                    json={"optical_path": optical_path, "sar_path": sar_path, "task": task, "parameters": parameters or {}}
                )
                resp.raise_for_status()
                data = resp.json()
                return ToolResult(
                    tool="sar_ml_fusion",
                    status="success",
                    result=data.get("result", {}),
                    artifacts=[ToolResultArtifact(**a) for a in data.get("artifacts", [])],
                    metadata={"model": "SAR-ML-Fusion", "endpoint": settings.SAR_FUSION_ENDPOINT}
                )
        except Exception as e:
            logger.error("sar_fusion_http_failed", error=str(e))
            return ToolResult(
                tool="sar_ml_fusion",
                status="error",
                error=f"SAR-ML-Fusion HTTP inference failed: {str(e)}",
                metadata={"model": "SAR-ML-Fusion"}
            )

    def _predict_mock(self, task: str) -> ToolResult:
        return ToolResult(
            tool="sar_ml_fusion",
            status="success",
            result={
                "fusion_mode": "feature_level_cross_attention",
                "joint_observations": [
                    {
                        "category": "built_up_structures",
                        "optical_evidence": "Spectral reflection indicates urban structures.",
                        "sar_evidence": "Strong double-bounce backscatter confirms dense metallic/concrete buildings.",
                        "confidence": 0.95
                    },
                    {
                        "category": "water_bodies",
                        "optical_evidence": "Low NIR reflectance water body signature.",
                        "sar_evidence": "Specular surface scattering yielding very low VV/VH backscatter.",
                        "confidence": 0.96
                    }
                ],
                "fused_class_distribution": {
                    "built_up": 38.0,
                    "water": 24.0,
                    "vegetation": 38.0
                }
            },
            artifacts=[],
            metadata={"model": "SAR-ML-Fusion", "mode": "mock"}
        )

    async def health_check(self) -> bool:
        if settings.MODEL_BACKEND == "mock":
            return True
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                health_url = settings.SAR_FUSION_ENDPOINT.replace("/predict", "/health")
                resp = await client.get(health_url)
                return resp.status_code == 200
        except Exception:
            return False
