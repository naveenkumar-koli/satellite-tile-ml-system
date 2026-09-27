import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple

from app.config import settings

@contextmanager
def get_db_cursor():
    """
    Context manager yielding a SQLite cursor in WAL mode.
    Guarantees transaction commit and connection closure.
    """
    conn = sqlite3.connect(str(settings.DATABASE_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    try:
        cursor = conn.cursor()
        yield cursor
        conn.commit()
    finally:
        conn.close()

def init_db():
    """
    Initialize SQLite schema with necessary indexes for analyst spatial & semantic queries.
    """
    with get_db_cursor() as cursor:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tiles (
            tile_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            image_hash TEXT UNIQUE NOT NULL,
            ingested_at TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            uncertainty_entropy REAL NOT NULL,
            review_required INTEGER NOT NULL,
            probabilities_json TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            image_path TEXT,
            bbox_min_lat REAL,
            bbox_min_lon REAL,
            bbox_max_lat REAL,
            bbox_max_lon REAL,
            analyst_verified_class TEXT,
            analyst_notes TEXT
        );
        """)
        
        # Indexes for fast querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_class ON tiles(predicted_class);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_confidence ON tiles(confidence);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_review ON tiles(review_required);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_ingested ON tiles(ingested_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tiles_hash ON tiles(image_hash);")

# Ensure database tables exist upon import
init_db()

def calculate_sha256(data: bytes) -> str:
    """Calculate SHA-256 hash of raw tile bytes for deduplication & provenance."""
    return hashlib.sha256(data).hexdigest()

def insert_tile_record(
    tile_id: str,
    filename: str,
    image_hash: str,
    predicted_class: str,
    confidence: float,
    uncertainty_entropy: float,
    review_required: bool,
    probabilities: Dict[str, float],
    latency_ms: float,
    image_path: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Dict[str, Any]:
    """
    Insert or update a tile classification record in SQLite.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    probs_json = json.dumps(probabilities)
    
    bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon = (None, None, None, None)
    if bbox:
        bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon = bbox

    with get_db_cursor() as cursor:
        cursor.execute("""
            INSERT INTO tiles (
                tile_id, filename, image_hash, ingested_at,
                predicted_class, confidence, uncertainty_entropy,
                review_required, probabilities_json, latency_ms,
                image_path, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_hash) DO UPDATE SET
                predicted_class = excluded.predicted_class,
                confidence = excluded.confidence,
                uncertainty_entropy = excluded.uncertainty_entropy,
                review_required = excluded.review_required,
                probabilities_json = excluded.probabilities_json,
                latency_ms = excluded.latency_ms,
                ingested_at = excluded.ingested_at
        """, (
            tile_id, filename, image_hash, now_utc,
            predicted_class, confidence, uncertainty_entropy,
            1 if review_required else 0, probs_json, latency_ms,
            image_path, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon
        ))

    return {
        "tile_id": tile_id,
        "filename": filename,
        "image_hash": image_hash,
        "ingested_at": now_utc,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "uncertainty_entropy": uncertainty_entropy,
        "review_required": review_required,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
        "image_path": image_path
    }

def get_tile_by_id(tile_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single tile record by ID."""
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM tiles WHERE tile_id = ?", (tile_id,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["probabilities"] = json.loads(res["probabilities_json"])
        res["review_required"] = bool(res["review_required"])
        return res

def query_tiles(
    class_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    review_required: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Flexible analyst query interface with filtering by land-use class, confidence bounds, and review flags.
    """
    query = "SELECT * FROM tiles WHERE 1=1"
    params: List[Any] = []

    if class_name:
        query += " AND predicted_class = ?"
        params.append(class_name)
    if min_confidence is not None:
        query += " AND confidence >= ?"
        params.append(min_confidence)
    if max_confidence is not None:
        query += " AND confidence <= ?"
        params.append(max_confidence)
    if review_required is not None:
        query += " AND review_required = ?"
        params.append(1 if review_required else 0)

    query += " ORDER BY ingested_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    results = []
    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        for row in cursor.fetchall():
            item = dict(row)
            item["probabilities"] = json.loads(item["probabilities_json"])
            item["review_required"] = bool(item["review_required"])
            results.append(item)
    return results

def get_system_stats() -> Dict[str, Any]:
    """
    Compute aggregate intelligence statistics for analysts & offline telemetry.
    """
    with get_db_cursor() as cursor:
        # Total tiles
        cursor.execute("SELECT COUNT(*) FROM tiles;")
        total_tiles = cursor.fetchone()[0]

        # Tiles requiring review
        cursor.execute("SELECT COUNT(*) FROM tiles WHERE review_required = 1;")
        tiles_requiring_review = cursor.fetchone()[0]

        # Average confidence & latency
        cursor.execute("SELECT AVG(confidence), AVG(latency_ms) FROM tiles;")
        avg_conf, avg_lat = cursor.fetchone()
        
        # Per class distribution
        cursor.execute("""
            SELECT predicted_class, COUNT(*), AVG(confidence)
            FROM tiles
            GROUP BY predicted_class
            ORDER BY COUNT(*) DESC;
        """)
        class_stats = {
            row[0]: {
                "count": row[1],
                "avg_confidence": round(row[2], 4) if row[2] else 0.0
            }
            for row in cursor.fetchall()
        }

    return {
        "total_tiles_ingested": total_tiles,
        "tiles_requiring_review": tiles_requiring_review,
        "review_percentage": round((tiles_requiring_review / total_tiles * 100), 2) if total_tiles > 0 else 0.0,
        "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
        "average_latency_ms": round(avg_lat, 2) if avg_lat else 0.0,
        "class_distribution": class_stats
    }

def update_analyst_feedback(tile_id: str, verified_class: str, notes: Optional[str] = None) -> bool:
    """Store human analyst verification/correction for active learning & drift evaluation."""
    with get_db_cursor() as cursor:
        cursor.execute("""
            UPDATE tiles
            SET analyst_verified_class = ?, analyst_notes = ?, review_required = 0
            WHERE tile_id = ?
        """, (verified_class, notes, tile_id))
        return cursor.rowcount > 0
