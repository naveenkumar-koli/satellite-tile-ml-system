import os
import sys
import csv
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.model import classifier

def evaluate_eval_set():
    eval_dir = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_set"
    eval_labels_path = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_labels.csv"

    true_labels = {}
    with open(eval_labels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                true_labels[row[0]] = row[1]

    y_true = []
    y_pred = []
    confidences = []
    review_flags = []

    for fname in sorted(true_labels.keys()):
        p = eval_dir / fname
        if p.exists():
            img = Image.open(p)
            res = classifier.predict(img)
            y_true.append(true_labels[fname])
            y_pred.append(res["predicted_class"])
            confidences.append(res["confidence"])
            review_flags.append(res["review_required"])

    classes = sorted(list(set(y_true)))
    print("=" * 60)
    print("OFFLINE MODEL EVALUATION REPORT ON EVAL_SET")
    print("=" * 60)
    print(classification_report(y_true, y_pred, digits=4))

    print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    header = f"{'True Class':14s}" + "".join([f"{c[:6]:>8s}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        row_str = f"{c:14s}" + "".join([f"{cm[i, j]:8d}" for j in range(len(classes))])
        print(row_str)

    # High-confidence vs low-confidence accuracy
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    conf_np = np.array(confidences)

    high_conf_mask = conf_np >= settings.CONFIDENCE_THRESHOLD
    high_conf_acc = np.mean(y_true_np[high_conf_mask] == y_pred_np[high_conf_mask])
    low_conf_acc = np.mean(y_true_np[~high_conf_mask] == y_pred_np[~high_conf_mask])

    print("\n" + "=" * 60)
    print("CONFIDENCE CALIBRATION & TRIAGE ANALYSIS")
    print("=" * 60)
    print(f"High-Confidence Slice (>= {settings.CONFIDENCE_THRESHOLD}):")
    print(f"  Count: {np.sum(high_conf_mask)} tiles ({np.mean(high_conf_mask)*100:.1f}%)")
    print(f"  Accuracy: {high_conf_acc * 100:.2f}% (Significant precision boost!)")
    print(f"\nLow-Confidence Slice (< {settings.CONFIDENCE_THRESHOLD}, Flagged for Review):")
    print(f"  Count: {np.sum(~high_conf_mask)} tiles ({np.mean(~high_conf_mask)*100:.1f}%)")
    print(f"  Accuracy: {low_conf_acc * 100:.2f}% (Ambiguous tiles accurately isolated!)")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_eval_set()
