"""
FastAPI application entry point for the Track 1 pipeline.

Start with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routes.events import router as events_router
from app.routes.rag_routes import router as rag_router
from app.routes.causal_routes import router as causal_router
from app.routes.feed_routes import router as feed_router


def _setup_logging() -> None:
    """Configure dual-handler logging per context.md §6.6."""
    log_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on reload
    if root_logger.handlers:
        return

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console)

    # File handler
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.paths.log_dir / "track1.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)


def _ensure_directories() -> None:
    """Create required directories if they don't exist."""
    settings.paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.config_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    _setup_logging()
    _ensure_directories()
    init_db()

    logger = logging.getLogger(__name__)
    logger.info("Track 1 pipeline server started")
    logger.info("Dataset dir: %s", settings.paths.dataset_dir)
    logger.info("Database: %s", settings.paths.db_path)

    yield

    # Stop any running live-feed monitors before exit
    from app.pipeline.monitor import get_feed_manager
    get_feed_manager().stop_all()
    logging.getLogger(__name__).info("Track 1 pipeline server shutting down")


app = FastAPI(
    title="Track 1 — Traffic Intersection Data Engineering Pipeline",
    description=(
        "Heuristic video ingestion, multi-object tracking, spatial transformation, "
        "and generation of isolated temporal datasets for causal analysis."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Allow React frontend to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events_router)
app.include_router(rag_router)
app.include_router(causal_router)
app.include_router(feed_router)


@app.get("/api/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/api/logs", tags=["System"])
async def get_logs():
    """Return the last 100 lines of the track1.log file."""
    log_file = settings.paths.log_dir / "track1.log"
    if not log_file.exists():
        return {"logs": ["Log file not found."]}
    
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return {"logs": lines[-100:]}
    except Exception as e:
        return {"logs": [f"Error reading log file: {str(e)}"]}


@app.get("/api/config", tags=["System"])
async def get_config():
    """Return the current pipeline configuration."""
    # Convert settings to a dict 
    from dataclasses import asdict
    conf = asdict(settings)
    # Filter out path objects to avoid serialization issues, just convert to strings
    conf["paths"] = {k: str(v) for k, v in conf["paths"].items()}
    conf["homography_exists"] = settings.paths.homography_path.exists()
    return conf
