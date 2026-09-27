import io
import uuid
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.model import classifier
from app.db import (
    init_db,
    calculate_sha256,
    insert_tile_record,
    get_tile_by_id,
    query_tiles,
    get_system_stats,
    update_analyst_feedback,
    get_db_cursor
)
from app.schemas import (
    IngestResponse,
    IngestPathRequest,
    TileRecord,
    QueryResponse,
    StatsResponse,
    FeedbackRequest,
    HealthResponse
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("galaxeye_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and check model
    logger.info("Initializing offline database...")
    init_db()
    
    if not classifier.is_loaded:
        logger.warning(
            f"Classifier not found at {settings.MODEL_PATH}. "
            "Please run `python scripts/train_model.py`."
        )
    else:
        logger.info(f"Loaded classifier with {len(classifier.classes)} classes.")
    yield
    # Shutdown
    logger.info("Service shutting down cleanly.")

app = FastAPI(
    title=settings.APP_NAME,
    description="Offline backend service for ingesting satellite image tiles, running local edge classification, and serving analyst queries.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards / GIS web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["System"])
def root():
    """Root endpoint welcoming the user and directing to interactive documentation."""
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "status": "operational",
        "documentation_url": "/docs",
        "endpoints": {
            "health": "/health",
            "ingest_tile": "POST /ingest",
            "ingest_by_path": "POST /ingest-path",
            "analyst_query": "GET /query",
            "system_statistics": "GET /stats",
            "analyst_feedback": "POST /tiles/{tile_id}/feedback"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and offline status check."""
    db_ok = False
    try:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT 1;").fetchone()
            db_ok = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")

    return {
        "status": "healthy" if (db_ok and classifier.is_loaded) else "degraded",
        "offline_mode": True,
        "model_loaded": classifier.is_loaded,
        "database_connected": db_ok,
        "classes_supported": settings.CLASSES,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD
    }

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def ingest_tile(file: UploadFile = File(...)):
    """
    Core Path Endpoint:
    Ingests an uploaded satellite image tile, validates it, computes SHA-256 hash,
    runs offline classification, and stores the results in SQLite.
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Offline classifier model is not loaded. Run train_model.py first."
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        # Compute checksum
        image_hash = calculate_sha256(content)
        tile_id = f"tile_{image_hash[:12]}"

        # Load image via PIL
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid or corrupted image format: {str(e)}"
            )

        # Cache/persist raw tile to local storage
        out_path = settings.STORAGE_DIR / f"{tile_id}_{file.filename}"
        with open(out_path, "wb") as f_out:
            f_out.write(content)

        # Run offline prediction
        pred = classifier.predict(image)

        # Persist to database
        record = insert_tile_record(
            tile_id=tile_id,
            filename=file.filename or "unknown.png",
            image_hash=image_hash,
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            uncertainty_entropy=pred["uncertainty_entropy"],
            review_required=pred["review_required"],
            probabilities=pred["probabilities"],
            latency_ms=pred["latency_ms"],
            image_path=str(out_path)
        )

        return record

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during tile ingestion")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/ingest-path", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
def ingest_tile_by_path(req: IngestPathRequest):
    """
    Alternative Ingest Endpoint for Batch / Local Pipelines:
    Ingests a satellite tile already stored on the local filesystem (e.g. from an edge sensor/camera buffer).
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Offline classifier model is not loaded."
        )

    tile_path = Path(req.filepath)
    if not tile_path.exists() or not tile_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tile file not found at path: {req.filepath}"
        )

    try:
        content = tile_path.read_bytes()
        image_hash = calculate_sha256(content)
        tile_id = req.tile_id or f"tile_{image_hash[:12]}"

        image = Image.open(io.BytesIO(content))
        image.load()

        pred = classifier.predict(image)

        bbox_tuple = tuple(req.bbox) if req.bbox and len(req.bbox) == 4 else None

        record = insert_tile_record(
            tile_id=tile_id,
            filename=tile_path.name,
            image_hash=image_hash,
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            uncertainty_entropy=pred["uncertainty_entropy"],
            review_required=pred["review_required"],
            probabilities=pred["probabilities"],
            latency_ms=pred["latency_ms"],
            image_path=str(tile_path),
            bbox=bbox_tuple
        )
        return record
    except Exception as e:
        logger.exception("Error ingesting tile from path")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/query", response_model=QueryResponse, tags=["Analyst Queries"])
def query_results(
    class_name: Optional[str] = Query(None, description="Filter by predicted land-use class (e.g., Forest, River, Residential)"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score filter"),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum confidence score filter"),
    review_required: Optional[bool] = Query(None, description="Filter tiles flagged for human analyst review"),
    limit: int = Query(50, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Analyst Query Endpoint:
    Allows an analyst to query, filter, and inspect classified satellite tiles offline.
    """
    results = query_tiles(
        class_name=class_name,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        review_required=review_required,
        limit=limit,
        offset=offset
    )
    return {
        "count": len(results),
        "results": results
    }

@app.get("/tiles/{tile_id}", response_model=TileRecord, tags=["Analyst Queries"])
def get_tile(tile_id: str):
    """Fetch details and full class probability breakdown for a specific tile."""
    record = get_tile_by_id(tile_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tile '{tile_id}' not found.")
    return record

@app.post("/tiles/{tile_id}/feedback", tags=["Human-in-the-Loop"])
def submit_feedback(tile_id: str, feedback: FeedbackRequest):
    """
    Allows an analyst to verify or correct a classification result.
    Useful for offline model calibration and auditing.
    """
    if feedback.verified_class not in settings.CLASSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown class '{feedback.verified_class}'. Must be one of {settings.CLASSES}"
        )

    updated = update_analyst_feedback(
        tile_id=tile_id,
        verified_class=feedback.verified_class,
        notes=feedback.notes
    )
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tile '{tile_id}' not found.")
    return {"message": "Analyst feedback recorded successfully.", "tile_id": tile_id}

@app.get("/stats", response_model=StatsResponse, tags=["Analytics & Telemetry"])
def system_statistics():
    """
    Aggregate telemetry and intelligence summary:
    Total tiles ingested, review backlog percentage, average confidence, latency, and class breakdown.
    """
    return get_system_stats()
