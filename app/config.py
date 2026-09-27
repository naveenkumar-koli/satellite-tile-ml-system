import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "GalaxEye Offline Tile Ingestion & Classification Service"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "artifacts_models" / "eurosat_classifier.joblib"
    DATABASE_PATH: Path = BASE_DIR / "satellite_tiles.db"
    STORAGE_DIR: Path = BASE_DIR / "storage" / "tiles"
    
    # Confidence threshold: predictions below this are flagged as requiring analyst review
    CONFIDENCE_THRESHOLD: float = 0.65
    
    # Supported EuroSAT land-use classes
    CLASSES: list[str] = [
        "AnnualCrop",
        "Forest",
        "Highway",
        "Industrial",
        "Residential",
        "River",
        "SeaLake"
    ]

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure storage directory exists
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
