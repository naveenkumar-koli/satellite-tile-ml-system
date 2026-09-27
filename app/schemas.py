from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class TilePrediction(BaseModel):
    predicted_class: str
    confidence: float
    uncertainty_entropy: float
    review_required: bool
    probabilities: Dict[str, float]
    latency_ms: float

class IngestResponse(TilePrediction):
    tile_id: str
    filename: str
    image_hash: str
    ingested_at: str
    image_path: Optional[str] = None
    bbox: Optional[List[float]] = None

class IngestPathRequest(BaseModel):
    filepath: str = Field(..., description="Absolute or relative local filepath to satellite tile image")
    tile_id: Optional[str] = Field(None, description="Optional custom identifier for tile")
    bbox: Optional[List[float]] = Field(None, description="Optional [min_lat, min_lon, max_lat, max_lon]")

class TileRecord(BaseModel):
    tile_id: str
    filename: str
    image_hash: str
    ingested_at: str
    predicted_class: str
    confidence: float
    uncertainty_entropy: float
    review_required: bool
    probabilities: Dict[str, float]
    latency_ms: float
    image_path: Optional[str] = None
    bbox_min_lat: Optional[float] = None
    bbox_min_lon: Optional[float] = None
    bbox_max_lat: Optional[float] = None
    bbox_max_lon: Optional[float] = None
    analyst_verified_class: Optional[str] = None
    analyst_notes: Optional[str] = None

class QueryResponse(BaseModel):
    count: int
    results: List[TileRecord]

class StatsResponse(BaseModel):
    total_tiles_ingested: int
    tiles_requiring_review: int
    review_percentage: float
    average_confidence: float
    average_latency_ms: float
    class_distribution: Dict[str, Any]

class FeedbackRequest(BaseModel):
    verified_class: str
    notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    offline_mode: bool
    model_loaded: bool
    database_connected: bool
    classes_supported: List[str]
    confidence_threshold: float
