# Problem-Solving & Operational Reasoning (Part 3)

**Author**: Senior Backend & ML Systems Engineer Candidate  
**Company**: GalaxEye — Backend Engineer, ML Systems Assignment  

---

### Question 1: Classifier Error Rate & Operational Viability

> **Prompt**: *Your classifier turns out to be wrong about 30% of the time. What do you do — and how would you even decide whether it’s “good enough” to be useful at all?*

#### 1. What I Would Do
1. **Deconstruct Aggregate Error into an Asymmetric Cost Matrix**:
   - An overall 30% error rate (70% accuracy) is an aggregate metric that hides critical operational reality. In satellite remote sensing, classification errors have wildly asymmetric consequences:
     - Misclassifying `AnnualCrop` as `Forest` has near-zero operational impact.
     - Misclassifying a `River` or flood corridor as `Highway` or `Residential` can lead to disaster response failure or navigation catastrophe.
   - I would immediately inspect the per-class confusion matrix (as demonstrated in our evaluation: errors are heavily concentrated in linear features like `Highway` vs `River` confusing with surrounding terrain).
2. **Implement Selective Classification (Confidence-Based Triage)**:
   - Rather than forcing an argmax prediction on every tile, we calibrate output probabilities and set a threshold (e.g., $p_{\max} \ge 0.65$).
   - **Empirical Evidence from Our Evaluation**: On the EuroSAT evaluation dataset, the high-confidence slice ($\ge 0.65$) achieves **97.86% accuracy** across 67% of tiles. The 30% error rate is almost entirely confined to the low-confidence 33% slice (which drops to 58.57% accuracy).
   - We automatically pass the 97.9% accurate tiles directly to downstream reports and route the remaining ambiguous 33% into an analyst review queue.
3. **Exploit Spatial Autocorrelation (Context Priors)**:
   - Satellite tiles do not exist in a vacuum; they form contiguous geographical swaths. We apply spatial neighborhood smoothing (e.g., Markov Random Fields or 3×3 majority voting). If an isolated tile is predicted as `River` surrounded entirely by 8 `Forest` tiles, spatial consensus can correct the prediction without retraining.
4. **Targeted Feature / Band Enrichment**:
   - Supplement RGB with spectral indices (e.g., Near-Infrared NDVI if multispectral bands are available, or Haralick GLCM texture to separate asphalt roads from water bodies).

#### 2. How to Decide Whether It’s "Good Enough" to Be Useful
- **Compare to the Human Baseline / Operational Alternative**:
  - If analysts currently have to manually pan through 20,000 raw tiles per day, a model that accurately triages 67% of the tiles with 98% accuracy reduces manual analyst workload by two-thirds. That is enormously valuable today, even with 30% total error.
- **Cost-Benefit Equation**:
  $$\text{Expected Value} = \sum_{i,j} P(Y=i, \hat{Y}=j) \cdot \text{Utility}(i, j) - \text{Cost}_{\text{Review}}$$
  If the operational savings of automated processing minus the cost of analyst verification of flagged tiles is positive and superior to manual scanning, the model is "good enough" for production deployment.
- **Service Level Objectives (SLOs)**:
  - We establish strict per-class SLOs: e.g., Recall $\ge 95\%$ on high-priority infrastructure, Precision $\ge 85\%$ on background land-use. If a model meets the target for critical classes, lower accuracy on ambiguous borderline classes is operationally acceptable.

---

### Question 2: Offline Observability & Health Verification

> **Prompt**: *This service runs offline, with no one watching it live. A month after deployment, how would you know it’s still working correctly?*

When deploying to air-gapped, isolated edge nodes (such as satellite payload computers or remote ground stations), there are no cloud dashboards, Prometheus pull endpoints, or Slack alerts. Verification must rely on **autonomous, self-contained local telemetry, deterministic golden canaries, and distribution baselines**.

#### 1. Embedded "Golden Canary" Benchmark (Ground Truth Checkpoint)
- The system embeds a suite of 20 immutable "golden tiles" with pre-verified ground truth directly into local storage.
- Every 24 hours (or after every 1,000 ingested tiles during an idle cycle), an automated cron task executes an end-to-end inference pass on these canary tiles.
- If:
  - Output predictions deviate from the golden labels,
  - Softmax probabilities drift by more than 5%,
  - Inference latency spikes beyond a 3-sigma bound,
  the system logs a critical hardware alert into `system_health.json` and trips a local health status from `HEALTHY` to `DEGRADED`.

#### 2. Input Data Distribution Monitoring (Data Drift)
- Physical sensors degrade: lenses accumulate dirt, optical sensors experience thermal drift, SAR sensors experience receiver gain changes, and solar angles change seasonally.
- The service maintains a rolling 30-day histogram of raw input telemetry:
  - Band pixel intensity distributions (mean, std, percentiles).
  - Rate of corrupted, unreadable, or all-black/all-white saturated tiles.
- If the Wasserstein distance between the current week's input distribution and the baseline training distribution exceeds an operational threshold, the system flags sensor calibration drift.

#### 3. Output Prediction Distribution Monitoring (Concept Drift)
- Over a large geographical region, land-use class distributions follow predictable macro baselines (e.g., 40% Forest, 30% Crop, 15% Water, 10% Residential, 5% Industrial).
- If the offline database records that `Industrial` suddenly accounts for 75% of all predictions over the past two weeks, this statistical anomaly signals that the classifier has collapsed or the input channel mapping has drifted.

#### 4. Hardware Watermarks & Storage Health
- Self-contained SQLite health metrics tracked via lightweight append-only logs:
  - Database file integrity verification (`PRAGMA integrity_check;`).
  - Storage consumption and disk watermark thresholds (ensuring WAL log files checkpoint cleanly without filling up disk).
  - RSS memory usage and per-tile processing latency percentiles ($p50, p95, p99$).

#### 5. Offline Audit Packager for Analysts
- When an analyst periodically visits the hardware or downlinks an archive:
  - Running a single CLI command (`python scripts/health_report.py`) reads the local telemetry log and outputs an audit report: canary test history, drift metrics, error logs, and SQLite database integrity status.

---

### Question 3: Root Cause Analysis for Inaccurate Stored Results

> **Prompt**: *Tiles are coming in fine, but the stored results look wrong. Walk us through how you’d find the cause — your actual steps, in order.*

I would execute a rigorous **upstream-to-downstream binary search**, isolating each boundary of the system:

```
[Raw Bytes on Disk] -> [Image Decoding] -> [Preprocessing/Channels] -> [Feature Tensor] -> [Model Logits] -> [Label Mapping] -> [DB Schema Insert] -> [Query/View]
```

#### Step 1: Inspect the Raw Tile on Disk (Sanity & Sensor Check)
- **Action**: Extract 5 specific tiles where the stored result was flagged as wrong. Render them visually and inspect file headers (`exiftool` / `file` command).
- **Check**: Did the upstream camera or satellite downlink change format? (e.g., 16-bit uint16 GeoTIFF written where the service expected 8-bit uint8 PNG, or a corrupted Bayer pattern).

#### Step 2: Check Color Channel Ordering & Normalization (The #1 ML Bug)
- **Action**: Log the decoded array shape, data type, and channel order:
  ```python
  arr = np.array(Image.open(tile_path))
  print(arr.shape, arr.dtype, arr.min(), arr.max())
  ```
- **Check**: Is there an RGB vs BGR channel inversion?
  - If OpenCV (`cv2.imread` reads BGR) was used upstream while the model was trained on PIL (`RGB`), Red and Blue channels are swapped. Clear blue water suddenly appears red (classified as Industrial/Residential), and green forests shift hue.
  - Check dynamic range: If image values are $[0, 1]$ float but the model expects $[0, 255]$ integer (or vice versa), the feature extractor produces complete garbage.

#### Step 3: Validate the Feature Extraction Tensor
- **Action**: Run `extract_features(tile)` directly on the offending tile and compare the resulting 94-dimensional vector against the training distribution statistics.
- **Check**: Are any features `NaN`, `Inf`, or extreme outliers? (e.g., a division by zero in the vegetation index proxy `(G - R) / (G + R)` if epsilon was missing).

#### Step 4: Isolate Model Output (Logits vs Probabilities)
- **Action**: Feed the feature vector directly into `clf.predict_proba()` in a standalone Python interpreter.
- **Check**:
  - Is the model producing a confident wrong prediction (e.g., 99% confident in `Industrial` for a `Forest` tile)? That indicates an unhandled visual domain (clouds, snow, glint).
  - Or is the distribution completely flat / uniform (~14% across all 7 classes)? A flat distribution indicates the input features lie completely outside the training manifold.

#### Step 5: Verify Class Label Encoder Index Alignment
- **Action**: Compare `clf.classes_` array with the label mapping used during database insertion:
  ```python
  top_idx = np.argmax(probs)
  predicted_class = clf.classes_[top_idx]
  ```
- **Check**: Silent index desynchronization! If the model artifact was trained with alphabetically sorted labels `['AnnualCrop', 'Forest', 'Highway', 'Industrial', ...]` but the database insertion logic assumed an integer mapping from a different dictionary, every prediction will be systematically shifted by 1 or 2 indices.

#### Step 6: Verify Database Write & SQL Parameter Binding
- **Action**: Query the raw database row via `sqlite3`:
  ```sql
  SELECT tile_id, predicted_class, confidence, probabilities_json, latency_ms FROM tiles WHERE tile_id = '...';
  ```
- **Check**: Positional argument slippage in the SQL `INSERT` statement! (e.g., if a parameter was added to `INSERT INTO tiles (...) VALUES (?, ?, ...)` and positional arguments shifted, `confidence` might be stored in the `latency_ms` column or `predicted_class` mapped incorrectly).

#### Step 7: Verify Analyst Query & Filtering Logic
- **Action**: Check the exact SQL query executed by the analyst's client.
- **Check**: Is the analyst querying with an un-indexed JOIN, an inverted `ORDER BY`, or an ambiguous timezone translation on `ingested_at`?

---

### Question 4: Weakest Link & Failure Mode Analysis

> **Prompt**: *What’s the weakest part of your design, and what would break it first?*

#### 1. The Weakest Part of the Design
**Spatial Isolation: Classifying 64×64 Tiles Without Context.**

In remote sensing, context is paramount. A 64×64 pixel tile covers an area of roughly $640 \times 640\text{ meters}$ at Sentinel-2's 10-meter ground sampling distance. 
- In complete isolation, an asphalt highway running through farmland looks almost identical to a dark river corridor.
- A cluster of rooftops in an industrial warehouse zone looks nearly identical to dense residential buildings.
- A flat expanse of sand or bare soil looks identical to an unplanted crop field.

Because each tile is classified as an independent entity, the model cannot utilize macro-scale topographic continuity (e.g., roads connect to other roads; rivers terminate in lakes; residential developments border transport arteries).

#### 2. What Would Break It First
**Atmospheric & Environmental Domain Shift (Clouds, Haze, Sun Glint, and Snow).**

The EuroSAT dataset is a curated benchmark composed of clean, cloud-free, top-of-atmosphere/bottom-of-atmosphere optical imagery taken under consistent European daylight conditions.

In real-world deployment on satellite hardware:
1. **Cloud Cover & Shadows**: Up to 60% of optical satellite imagery at any given time contains clouds or cloud shadows. A thick white cloud over a forest will saturate the RGB channels and cause the model to misclassify it as `Industrial` (high reflectance concrete) with high confidence. A cloud shadow will be misclassified as `SeaLake` or `River`.
2. **Solar Elevation / Seasonal Illumination**: Winter passes with low sun angles cast elongated shadows, reducing overall pixel radiance and shifting color distributions beyond the training envelope.
3. **Sensor Radiometric Shift**: Multi-spectral sensors experience calibration decay over time. Without automated cloud-masking (e.g. SCL - Scene Classification Layer or blue-band Cirrus thresholds) and radiometric normalization, raw uncalibrated tiles will rapidly degrade classifier accuracy.

#### 3. How I Would Strengthen It in Phase 2
1. **Add a Dedicated "Cloud / Shadow / No-Data" Triage Filter**: Prepend a fast radiometric rule-based filter that checks saturation, brightness variance, and cloud likelihood before running the land-use classifier.
2. **Incorporate Multi-Tile Spatial Windowing**: Allow the model to ingest adjacent neighboring tiles (a $3 \times 3$ tile context window) or apply Conditional Random Field (CRF) post-processing to enforce geographical continuity.
3. **Disk Retention & FIFO Pruning Policy**: Implement a bounded ring-buffer for cached tile images on disk with strict high-watermark triggers to prevent edge storage exhaustion.
