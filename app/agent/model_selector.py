from app.core.logging import logger
from app.models.registry import model_registry
from app.schemas.analysis import QueryAnalysis
from app.schemas.images import ImageInputConfig
from app.schemas.models import ModelSelection


class ModelSelector:
    """Capability-based model selector.
    Chooses specialist models based on task compatibility, input modalities, image count, and availability.
    """

    async def select_model(
        self,
        analysis: QueryAnalysis,
        input_config: ImageInputConfig
    ) -> ModelSelection:
        modalities = [img.modality for img in input_config.images]
        image_count = input_config.image_count

        # Query registry for capable models
        candidates = model_registry.find_capable_models(
            task=analysis.task,
            modalities=modalities,
            image_count=image_count
        )

        if not candidates:
            # Fallback lookup by intent
            candidates = model_registry.find_capable_models(
                task="any",
                modalities=modalities,
                image_count=image_count
            )

        if not candidates:
            # Fall back to GeoChat as default single image VLM or ChangeChat for pair
            default_model = "changechat" if image_count >= 2 else "geochat"
            return ModelSelection(
                selected_model=default_model,
                task=analysis.task,
                reasoning=f"Default fallback selection for image_count={image_count}.",
                confidence=0.60
            )

        # Select candidate with highest suitability
        selected = candidates[0]
        fallbacks = [c.name for c in candidates[1:]]

        reasoning = (
            f"Selected {selected.name} based on task compatibility ({analysis.task}), "
            f"supported modalities ({modalities}), and image count ({image_count})."
        )

        logger.info("model_selected", selected=selected.name, task=analysis.task)

        return ModelSelection(
            selected_model=selected.name,
            task=analysis.task,
            reasoning=reasoning,
            confidence=0.92,
            fallback_models=fallbacks
        )


model_selector = ModelSelector()
