# GalaxEye — Offline Satellite Tile Classification & Query Service
**Backend Engineer, ML Systems — Take-Home Assignment Solution**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Offline First](https://img.shields.io/badge/Environment-100%25%20Offline%20CPU-brightgreen.svg)]()

This repository contains the complete implementation and design submission for GalaxEye's **Backend Engineer, ML Systems** take-home assignment.

---

## Deliverables Index

- **[Part 1: Comprehensive Design Note](file:///c:/Users/nitro/satalite/DESIGN_NOTE.md)** — Architectural design, tile lifecycle, trade-offs (uncertainty handling, storage, query semantics), assumptions, and critical stakeholder questions.
- **[Part 2: Working Slice (Code & Quickstart)](#quickstart--running-the-service)** — Production-grade FastAPI backend with zero cloud dependencies, offline CPU inference (<2ms/tile), SQLite WAL persistence, and analyst query API.
- **[Part 3: Problem-Solving & Operational Reasoning](file:///c:/Users/nitro/satalite/PROBLEM_SOLVING.md)** — Deep technical responses to the 4 operational scenario questions (error handling, offline observability, root-cause debugging, and failure-mode analysis).

---

## 1. System Overview

Designed for isolated satellite payloads or air-gapped ground station hardware, this service operates **100% offline** without any internet access or hosted cloud APIs.

```
                      +---------------------------------------+
                      |         AIR-GAPPED EDGE NODE          |
                      +---------------------------------------+
                                          |
   [Raw Satellite Tile]                   |
            |                             v
            +------------->  [POST /ingest or Ingest Watcher]
                                          |
                                          v
                             [Tile Validation & SHA-256]
                                          |
                                          v
                             [Spectral-Spatial Extractor]
                                          |
                                          v
                              [Offline CPU Classifier]
                                          |
                    +---------------------+---------------------+
                    |                                           |
                    v                                           v
       (High Conf: p >= 0.65)                      (Low Conf: p < 0.65)
        97.9% Accurate Slice                        Flagged for Review
                    |                                           |
                    +---------------------+---------------------+
                                          |
                                          v
                             [SQLite WAL Persistence]
                             (Indexed Metadata & JSON)
                                          |
                                          v
                             [Analyst Query & Stats API]
                              (/query, /stats, /feedback)
```

---

## 2. Key Technical Highlights

1. **Deterministic Edge Feature Extraction**: Computes 94-dimensional spectral-spatial representations (channel percentiles, vegetation/water index proxies, spatial grid pooling, and edge gradients) in **< 1.5ms on CPU**.
2. **Offline Random Forest Model**: Compact 0.58 MB serialized bundle with zero GPU or heavyweight deep-learning runtime dependencies.
3. **Selective Classification (Triage Filter)**:
   - Tiles with confidence $\ge 0.65$ achieve **97.86% accuracy** (67% of tiles).
   - Ambiguous tiles ($< 0.65$) drop to 58.57% accuracy and are automatically flagged with `review_required = True`.
4. **SQLite with Write-Ahead Logging (WAL)**: Zero-daemon embedded storage supporting concurrent reads during writes, indexed spatial bounding boxes, and single-file portability.
5. **Analyst Workflow**: Multi-dimensional filtering by land-use class, confidence bounds, human review flag, and interactive feedback endpoint.

---

## 3. Quickstart & Running the Service

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train & Verify Offline Model
The model artifact is already pre-trained and bundled at `artifacts_models/eurosat_classifier.joblib`. To re-train from candidate tiles:
```bash
python scripts/train_model.py
```

### Step 3: Run the Service
Start the FastAPI server on localhost port 8000:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 4. Endpoints & API Reference

Interactive Swagger documentation is available offline at `http://localhost:8000/docs`.

### 1. Ingest a Tile (Multipart Upload)
```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@be-mlsys-assignment-dataset/eval_set/tile_001.png"
```
**Response (201 Created):**
```json
{
  "tile_id": "tile_c7bede55c6fe",
  "filename": "tile_001.png",
  "image_hash": "c7bede55c6fe976ab1b2939d44543962dd782d891913ed8c1bb5b2efd719ec7b",
  "ingested_at": "2026-09-27T18:24:28.749545+00:00",
  "predicted_class": "Forest",
  "confidence": 1.0,
  "uncertainty_entropy": 0.0,
  "review_required": false,
  "probabilities": {
    "AnnualCrop": 0.0,
    "Forest": 1.0,
    "Highway": 0.0,
    "Industrial": 0.0,
    "Residential": 0.0,
    "River": 0.0,
    "SeaLake": 0.0
  },
  "latency_ms": 38.5,
  "image_path": "storage/tiles/tile_c7bede55c6fe_tile_001.png"
}
```

### 2. Ingest via Local Filesystem Path (Batch/Buffer Mode)
```bash
curl -X POST "http://localhost:8000/ingest-path" \
     -H "Content-Type: application/json" \
     -d "{\"filepath\": \"be-mlsys-assignment-dataset/eval_set/tile_070.png\"}"
```

### 3. Query Results (Analyst Search)
```bash
# Query all high-confidence Forest tiles
curl "http://localhost:8000/query?class_name=Forest&min_confidence=0.80&limit=10"

# Query tiles requiring human review
curl "http://localhost:8000/query?review_required=true&limit=20"
```

### 4. Fetch System Telemetry & Statistics
```bash
curl "http://localhost:8000/stats"
```
**Response:**
```json
{
  "total_tiles_ingested": 210,
  "tiles_requiring_review": 70,
  "review_percentage": 33.33,
  "average_confidence": 0.7683,
  "average_latency_ms": 42.15,
  "class_distribution": {
    "AnnualCrop": {"count": 33, "avg_confidence": 0.7349},
    "Residential": {"count": 33, "avg_confidence": 0.7614},
    "River": {"count": 32, "avg_confidence": 0.5862},
    "Industrial": {"count": 31, "avg_confidence": 0.8855},
    "Forest": {"count": 29, "avg_confidence": 0.9074},
    "SeaLake": {"count": 29, "avg_confidence": 0.9369},
    "Highway": {"count": 23, "avg_confidence": 0.5336}
  }
}
```

### 5. Submit Analyst Ground-Truth Feedback
```bash
curl -X POST "http://localhost:8000/tiles/tile_c7bede55c6fe/feedback" \
     -H "Content-Type: application/json" \
     -d "{\"verified_class\": \"Forest\", \"notes\": \"Verified clear canopy\"}"
```

---

## 5. Benchmarking & Verification Scripts

### Run Comprehensive Automated Service Test
```bash
python scripts/test_service.py
```

### Ingest Full Evaluation Dataset (210 Tiles)
```bash
python scripts/batch_ingest.py
```

### Run Confusion Matrix & Triage Calibration Analysis
```bash
python scripts/evaluate.py
```

#### Evaluation Summary on Ground Truth `eval_set`:
```
              precision    recall  f1-score   support

  AnnualCrop     0.8182    0.9000    0.8571        30
      Forest     1.0000    0.9667    0.9831        30
     Highway     0.6087    0.4667    0.5283        30
  Industrial     0.9032    0.9333    0.9180        30
 Residential     0.9091    1.0000    0.9524        30
       River     0.6562    0.7000    0.6774        30
     SeaLake     1.0000    0.9667    0.9831        30

    accuracy                         0.8476       210
   macro avg     0.8422    0.8476    0.8428       210
```

---

## 6. Project Directory Layout

```
satalite/
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration (db path, model path, thresholds)
│   ├── db.py               # SQLite WAL connection manager, schema, queries & stats
│   ├── model.py            # Feature extractor & TileClassifier inference engine
│   ├── schemas.py          # Pydantic request/response validation schemas
│   └── main.py             # FastAPI service endpoints (/ingest, /query, /stats, /feedback)
├── scripts/
│   ├── train_model.py      # Trains and bundles offline model from candidate_tiles
│   ├── batch_ingest.py     # Batch ingestion runner & performance benchmark
│   ├── evaluate.py         # Confusion matrix and confidence calibration analysis
│   └── test_service.py     # Automated end-to-end service test suite
├── artifacts_models/
│   └── eurosat_classifier.joblib  # Serialized model artifact (0.58 MB)
├── be-mlsys-assignment-dataset/   # EuroSAT satellite dataset
│   ├── candidate_tiles/
│   ├── eval_set/
│   └── eval_labels.csv
├── DESIGN_NOTE.md          # Part 1: System Design Note (Architecture, Trade-offs)
├── PROBLEM_SOLVING.md      # Part 3: Operational Problem-Solving Scenarios
└── README.md               # Part 2: Quickstart, API Documentation & Benchmarks
```
