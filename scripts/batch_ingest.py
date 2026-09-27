import os
import sys
import csv
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.model import classifier
from app.db import (
    init_db,
    calculate_sha256,
    insert_tile_record,
    get_system_stats
)

def run_batch_ingestion():
    print("=" * 65)
    print("GalaxEye Offline Tile Ingestion & Evaluation Benchmark")
    print("=" * 65)

    eval_dir = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_set"
    eval_labels_path = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_labels.csv"

    if not eval_dir.exists():
        print(f"Error: eval_set not found at {eval_dir}")
        return

    # Load ground truth labels
    ground_truth = {}
    if eval_labels_path.exists():
        with open(eval_labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    ground_truth[row[0]] = row[1]

    tiles = sorted(list(eval_dir.glob("*.png")))
    print(f"Found {len(tiles)} evaluation tiles to ingest.")
    print(f"Confidence threshold for analyst review: {settings.CONFIDENCE_THRESHOLD}")
    print("\nIngesting tiles...")

    t_start = time.perf_counter()
    correct = 0
    total = len(tiles)
    review_count = 0
    latencies = []

    for idx, tile_path in enumerate(tiles, 1):
        content = tile_path.read_bytes()
        image_hash = calculate_sha256(content)
        tile_id = f"eval_{tile_path.stem}"

        image = Image.open(tile_path)
        pred = classifier.predict(image)
        latencies.append(pred["latency_ms"])

        if pred["review_required"]:
            review_count += 1

        true_label = ground_truth.get(tile_path.name)
        if true_label and true_label == pred["predicted_class"]:
            correct += 1

        insert_tile_record(
            tile_id=tile_id,
            filename=tile_path.name,
            image_hash=image_hash,
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence"],
            uncertainty_entropy=pred["uncertainty_entropy"],
            review_required=pred["review_required"],
            probabilities=pred["probabilities"],
            latency_ms=pred["latency_ms"],
            image_path=str(tile_path)
        )

        if idx % 50 == 0 or idx == total:
            print(f"  [{idx:3d}/{total}] Processed... Current avg latency: {sum(latencies)/len(latencies):.2f}ms")

    total_time = time.perf_counter() - t_start
    throughput = total / total_time

    print("\n" + "=" * 65)
    print("BATCH INGESTION & OFFLINE PIPELINE SUMMARY")
    print("=" * 65)
    print(f"Total tiles ingested       : {total}")
    print(f"Total elapsed time         : {total_time:.2f} seconds")
    print(f"Throughput                 : {throughput:.1f} tiles/second (Single-threaded CPU)")
    print(f"Mean inference latency     : {sum(latencies)/len(latencies):.2f} ms")
    if ground_truth:
        print(f"Ground Truth Accuracy      : {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"Flagged for Review (<{settings.CONFIDENCE_THRESHOLD}): {review_count}/{total} ({review_count/total*100:.1f}%)")
    
    print("\nDatabase Aggregate Statistics:")
    stats = get_system_stats()
    print(f"  - Total records in DB    : {stats['total_tiles_ingested']}")
    print(f"  - Class Breakdown        :")
    for cls, data in stats["class_distribution"].items():
        print(f"    * {cls:12s}: {data['count']:3d} tiles (avg confidence: {data['avg_confidence']:.2%})")

    print("=" * 65)
    print("All tiles successfully stored and indexed in SQLite.")

if __name__ == "__main__":
    run_batch_ingestion()
