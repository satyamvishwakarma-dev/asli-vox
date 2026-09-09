"""FastAPI main application entrypoint mounting routes, static assets, and WebSocket."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from asli_vox.config import settings
from asli_vox.utils.logger import logger
from asli_vox.api.routes import router as routes_router
from asli_vox.api.websocket import router as ws_router
from asli_vox.api.schemas import HealthResponse
from asli_vox.models.lightgbm_baseline import is_model_loaded
from asli_vox.features.embeddings import is_embeddings_enabled


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")
    logger.info(f"LightGBM model loaded: {is_model_loaded()}")
    logger.info(f"Heavy XLSR embeddings enabled: {is_embeddings_enabled()}")
    logger.info(f"Frontend directory: {settings.FRONTEND_DIR.resolve()}")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Real-Time AI Voice Deepfake & Spoof Detection (SIH 2026)",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware for hackathon demo compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API and WebSocket routers
app.include_router(routes_router)
app.include_router(ws_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health status check."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=is_model_loaded(),
        use_embeddings=is_embeddings_enabled(),
    )


# Mount static assets
frontend_path = settings.FRONTEND_DIR
if frontend_path.exists():
    # Mount /static for assets referenced as /static/... in HTML
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the frontend single page app."""
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend index.html not found"}
else:
    logger.warning(f"Frontend directory not found at {frontend_path}")
