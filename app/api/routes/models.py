from fastapi import APIRouter
from app.models.registry import model_registry
from app.schemas.models import ModelStatus

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("", response_model=list[ModelStatus])
async def list_registered_models():
    """Get list of registered specialist models, their health statuses, and capabilities."""
    return await model_registry.get_all_statuses()
