from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.models import router as models_router
from app.api.routes.reports import router as reports_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.models.init_registry import register_default_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize structured logging
    setup_logging(settings.LOG_LEVEL)

    # Register default specialist model adapters
    register_default_models()

    yield


app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Multimodal Remote-Sensing Intelligence Platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/swagger",
    redoc_url="/redoc",
)

# Enable CORS for React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routes
app.include_router(health_router)
app.include_router(models_router)
app.include_router(analyze_router)
app.include_router(jobs_router)
app.include_router(reports_router)


@app.get("/")
async def root():
    return {
        "title": "SatQuery AI Backend",
        "docs": "/swagger",
        "health": "/health",
        "models": "/api/v1/models"
    }
