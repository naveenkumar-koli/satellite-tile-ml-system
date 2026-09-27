# System Design Note: Offline Satellite Tile Ingestion & Classification Service
  
**Target Environment**: Isolated, Air-Gapped Edge Hardware / Ground Station Node (Zero Internet Access)  
**Task**: GalaxEye ML Systems Take-Home Assignment — Part 1  

---

## 1. Executive Summary & Context

GalaxEye operates in an environment where multi-sensor satellite imagery (SAR, optical, multispectral) must be transformed into actionable intelligence directly at the edge—either onboard satellite payloads or on ruggedized, isolated ground hardware with zero internet connectivity.

This document presents the architecture, engineering trade-offs, storage strategy, and edge considerations for a production-grade, offline-first tile classification and query service.

```
 +-----------------------------------------------------------------------------------+
 |                              OFFLINE EDGE HARDWARE                                |
 |                                                                                   |
 |  +--------------------+      +--------------------+      +---------------------+  |
 |  | Raw Tile Ingestion | ---> | Tile Validation &  | ---> | Offline Feature     |  |
 |  | (Upload / Watcher) |      | SHA-256 Checksum   |      | Extraction Pipeline |  |
 |  +--------------------+      +--------------------+      +----------+----------+  |
 |                                                                     |             |
 |                                                                     v             |
 |  +--------------------+      +--------------------+      +---------------------+  |
 |  | Analyst Query &    | <--- | SQLite Persistence | <--- | Offline Classifier  |  |
 |  | Telemetry API      |      | (WAL Mode + Index) |      | & Uncertainty Estim |  |
 |  +--------------------+      +--------------------+      +---------------------+  |
 +-----------------------------------------------------------------------------------+
```

---

## 2. End-to-End System Architecture & Tile Lifecycle

The system is decoupled into five distinct, lightweight components designed to operate deterministically on CPU-constrained offline hardware.

### 2.1 Component 1: Ingestion Gateway & Ingestion Watcher
- **Dual Ingestion Paths**:
  1. `POST /ingest`: HTTP multipart endpoint for interactive or network-streamed tile submission over local LAN.
  2. `POST /ingest-path` & Directory Watcher: Direct filesystem ingestion for tiles dumped by satellite telemetry downlinks or local camera buffer disks, avoiding redundant HTTP serialization overhead.
- **Zero-Trust Validation**: Every incoming file undergoes strict header inspection (magic bytes verification) and dimensional checks (e.g., EuroSAT 64×64 RGB tiles) before entering the ML pipeline.
- **Content-Addressable Deduplication**: Computes a SHA-256 digest of the raw tile bytes. If a tile was already ingested, it updates metadata idempotently without corrupting historical logs.

### 2.2 Component 2: Deterministic Preprocessing & Feature Extraction
- **Zero Hosted / Cloud Dependencies**: Operates entirely in-memory using localized vectorized NumPy operations.
- **Spectral-Spatial Representation**:
  - *Spectral Band Moments*: Per-channel mean, variance, and 10th/90th percentiles across Red, Green, and Blue bands.
  - *Remote Sensing Index Proxies*: Visible Atmospherically Resistant Index proxy (`(G - R) / (G + R + eps)`) for vegetation and Blue-to-Red band ratios for aquatic bodies.
  - *Spatial Grid Pooling*: 2×2 sub-quadrant spatial statistics to preserve spatial orientation (e.g., linear roads or rivers crossing specific quadrants).
  - *Texture & Gradient Proxies*: First-order difference gradients capturing urban density and edge frequency.
- **Performance**: Feature extraction takes **< 1.5 milliseconds per tile on standard CPU**.

### 2.3 Component 3: Offline Inference & Uncertainty Quantification
- **Model Architecture**: Calibrated ensemble classifier (Random Forest / Histogram Gradient Boosting) trained on EuroSAT domain data. Serialized as a lightweight artifact (`0.58 MB`), requiring zero network access to initialize.
- **Inference with Selective Classification (Rejection Option)**:
  - In high-stakes remote sensing, a silent wrong prediction is far more dangerous than an explicit abstention.
  - The model outputs:
    1. `predicted_class`: Argmax class label.
    2. `confidence`: Top probability $p_{\max} = \max_i p_i$.
    3. `uncertainty_entropy`: Shannon entropy $H(p) = -\sum_{i=1}^C p_i \log_2(p_i)$ measuring prediction ambiguity.
    4. `review_required`: Automated triage flag triggered when $p_{\max} < 0.65$ or entropy is high.

### 2.4 Component 4: Spatial & Metadata Persistence Layer
- **Storage Choice**: Single-file **SQLite** configured in **Write-Ahead Logging (WAL)** mode.
- **Schema Design**:
  - `tile_id`: Unique primary key (`tile_<sha256[:12]>`).
  - `filename`: Original tile identifier.
  - `image_hash`: SHA-256 unique digest.
  - `ingested_at`: ISO-8601 UTC timestamp.
  - `predicted_class`: Predicted land-use label.
  - `confidence`: Confidence score (float, 0.0 to 1.0).
  - `uncertainty_entropy`: Measure of prediction dispersion.
  - `review_required`: Boolean flag (1/0) for human-in-the-loop review.
  - `probabilities_json`: Serialized distribution across all classes for retrospective calibration.
  - `latency_ms`: Processing latency for edge telemetry.
  - `image_path`: Relative or absolute local filesystem path to the stored tile.
  - `bbox_min_lat`, `bbox_min_lon`, `bbox_max_lat`, `bbox_max_lon`: Geospatial bounding coordinates (WGS84).
  - `analyst_verified_class`, `analyst_notes`: Audit columns for analyst ground-truthing.
- **B-Tree Indexes**: Created on `predicted_class`, `confidence`, `review_required`, `ingested_at`, and `image_hash` to ensure sub-millisecond query responses across hundreds of thousands of tiles.

### 2.5 Component 5: Analyst Query & Intelligence Interface
- **RESTful Local API**:
  - `GET /query`: Multi-dimensional filtering by class, confidence intervals, review status, and pagination.
  - `GET /tiles/{tile_id}`: Deep inspection of a tile's complete probability distribution.
  - `POST /tiles/{tile_id}/feedback`: Analyst verification endpoint for active learning and model calibration.
  - `GET /stats`: Aggregate telemetry reporting class distributions, average confidence, latency, and review queue backlog.

---

## 3. Real Decisions & Architectural Trade-offs

| Engineering Decision | Options Considered | Decision Taken | Rationale & Trade-off |
| :--- | :--- | :--- | :--- |
| **Handling Low Confidence** | (A) Hard argmax output<br>(B) Reject & discard<br>(C) Selective classification with triage flag | **Option C: Selective Classification with `review_required`** | Satellite land-use analysis has asymmetric error costs (e.g., misclassifying water/forest vs crop). Discarding tiles causes spatial blindspots. Hard argmax hides errors. Selective triage routes high-confidence tiles (98% accuracy) directly to reports while queueing ambiguous tiles (58% accuracy) for analyst verification. |
| **Persistence Engine** | (A) PostgreSQL + PostGIS<br>(B) Flat GeoJSON / Parquet<br>(C) SQLite in WAL mode | **Option C: SQLite in WAL mode** | PostgreSQL requires background daemons, system services, and memory overhead unsuited for isolated/embedded devices. Flat files lack concurrent indexing and atomic ACID writes. SQLite WAL mode provides zero-daemon embedded persistence, concurrent reads during writes, sub-millisecond indexed lookups, and single-file air-gap portability. |
| **Image Data Storage** | (A) Blob inside DB<br>(B) Filesystem reference with DB pointer<br>(C) Store only features | **Option B: Local filesystem cache + DB URI** | Storing binary images in SQLite bloats the database file, degrading page cache performance. Storing only features prevents visual audit by analysts. Storing raw tiles in a structured content-addressed directory (`storage/tiles/<hash>.png`) keeps the database lightweight and fast. |
| **Query Semantics** | (A) Simple class string matching<br>(B) Multi-dimensional spatial & confidence filtering | **Option B: Multi-dimensional query engine** | Analysts do not just ask "show me forests". They ask: "Show me all high-confidence residential areas," "Show me low-confidence water tiles near river boundaries," and "Give me the review queue for yesterday's swath." |
| **Model Footprint** | (A) Large Vision Transformer (ViT)<br>(B) Heavy ResNet50 PyTorch<br>(C) Compact CPU ensemble / feature pipeline | **Option C: Compact CPU ensemble (0.58 MB)** | ViT and heavy CNNs require GPUs or significant CPU RAM/thermal overhead (>500MB weights, >200ms latency). The compact spectral-spatial pipeline executes in ~40ms on a single CPU core, fits in under 1 MB of disk space, and achieves **84.8% overall accuracy (and 97.9% high-confidence accuracy)**. |

---

## 4. Key Assumptions & Questions for GalaxEye Stakeholders

### Assumptions Made
1. **Isolated Operating Environment**: Hardware is completely air-gapped. Dependencies, weights, and runtimes must be bundled in advance (e.g. self-contained virtualenv or Docker image).
2. **Sensor Input**: The provided dataset represents Sentinel-2 optical RGB tiles (64×64 pixels). Tiles are ingested either in real-time batches from telemetry buffers or via local analyst upload.
3. **Analyst Workflow**: Analysts access the system locally via an offline browser, CLI, or local GIS client (e.g., QGIS reading SQLite/GeoPackage).
4. **Hardware Profile**: Target device is CPU-based with standard x86_64 or ARM64 architecture, standard POSIX filesystem, and restricted RAM/storage.

### Questions for GalaxEye Engineers & Product Team
1. **Sensor Modality & Multi-Sensor Fusion**:
   * *Question*: EuroSAT is optical RGB. GalaxEye specializes in SAR (Synthetic Aperture Radar) and multi-sensor imagery. Will the production pipeline ingest SAR backscatter (e.g. VV/VH polarizations), optical, or fused multi-modal data?
   * *Why it matters*: SAR imagery suffers from speckle noise and requires specialized radiometry (terrain flattening, Lee filters) and complex-valued neural network architectures or polarization ratio features.
2. **Tile Geometry & Spatial Referencing**:
   * *Question*: How are tiles georeferenced? Are tiles pre-cut into standard Slippy map / Web Mercator tiles (XYZ), or raw UTM satellite swath scenes with RPC (Rational Polynomial Coefficients)?
   * *Why it matters*: Determines whether we store spatial coordinates as bounding box floats or integrate `SpatiaLite` / R-Tree spatial indexing for polygon intersection queries.
3. **Deployment Hardware & SWaP Constraints**:
   * *Question*: What is the exact deployment target? Is this running on an onboard satellite payload computer (e.g., Unibap, Nvidia Jetson Orin Nano, Xilinx FPGA) or a ruggedized ground station laptop?
   * *Why it matters*: Drives memory allocations, quantization requirements (INT8/FP16 ONNX Runtime), thermal throttling limits, and watchdog process management.
4. **Data Offload & Analyst Hand-off**:
   * *Question*: How does the analyst consume results? Does the offline node produce a single exported archive (e.g., zipped SQLite database + GeoTIFF report) for air-gap transfer via encrypted USB, or does it feed a local LAN dashboard?
