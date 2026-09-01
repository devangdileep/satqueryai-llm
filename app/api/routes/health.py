from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "satquery-backend",
        "mode": settings.MODEL_BACKEND,
        "llm_provider": settings.LLM_PROVIDER
    }
