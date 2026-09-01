from typing import Optional
from app.core.logging import logger
from app.models.base import RemoteSensingModel
from app.schemas.models import ModelCapability, ModelStatus


class ModelRegistry:
    """Central registry of specialist remote-sensing models.
    Single source of truth for model capabilities and health status.
    """

    def __init__(self):
        self._models: dict[str, RemoteSensingModel] = {}

    def register(self, model: RemoteSensingModel) -> None:
        """Register a specialist model instance."""
        self._models[model.name.lower()] = model
        logger.info("model_registered", model_name=model.name, type=model.capabilities.type)

    def get_model(self, name: str) -> Optional[RemoteSensingModel]:
        """Get model by name (case-insensitive)."""
        return self._models.get(name.lower())

    def list_models(self) -> list[RemoteSensingModel]:
        """Return list of all registered models."""
        return list(self._models.values())

    async def get_all_statuses(self) -> list[ModelStatus]:
        """Check health of all registered models and return their statuses."""
        statuses = []
        for name, model in self._models.items():
            healthy = await model.health_check()
            statuses.append(
                ModelStatus(
                    name=model.name,
                    status="available" if healthy else "unavailable",
                    capabilities=model.capabilities,
                    healthy=healthy,
                    reason=None if healthy else "Backend service or GPU worker unavailable",
                )
            )
        return statuses

    def find_capable_models(
        self,
        task: str,
        modalities: list[str],
        image_count: int = 1
    ) -> list[RemoteSensingModel]:
        """Find models capable of executing the requested task and input configuration.
        Selection is strictly capability-based.
        """
        candidates = []
        for name, model in self._models.items():
            cap = model.capabilities

            # 1. Check if model supports the requested task
            task_match = any(t.lower() == task.lower() or task.lower() in t.lower() for t in cap.tasks)
            if not task_match and task.lower() not in ["any", "all"]:
                continue

            # 2. Check image count requirement
            if cap.requires_images > image_count:
                continue

            # 3. Check modality requirements if specified
            if cap.requires_modalities:
                req_met = all(m in modalities for m in cap.requires_modalities)
                if not req_met:
                    continue

            # 4. Check supported modalities
            modality_match = any(m in cap.supported_modalities for m in modalities) or "all" in cap.supported_modalities
            if not modality_match and len(modalities) > 0:
                continue

            candidates.append(model)

        logger.info(
            "model_candidates_found",
            task=task,
            modalities=modalities,
            image_count=image_count,
            candidates=[m.name for m in candidates]
        )
        return candidates


# Global singleton instance
model_registry = ModelRegistry()
