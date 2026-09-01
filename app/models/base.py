from abc import ABC, abstractmethod
from typing import Any
from app.schemas.models import ModelCapability
from app.schemas.tools import ToolResult


class RemoteSensingModel(ABC):
    """Abstract base class for all remote sensing model adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the model (e.g. GeoChat, ChangeChat, Prithvi, SAR-ML-Fusion)."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapability:
        """Model capabilities specification."""
        pass

    @abstractmethod
    async def predict(
        self,
        inputs: dict[str, Any],
        task: str,
        parameters: dict[str, Any] | None = None
    ) -> ToolResult:
        """Run prediction/inference using the model adapter."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model backend service/GPU is available."""
        pass
