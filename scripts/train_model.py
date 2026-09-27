import os
import sys
import csv
import time
from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to sys.path so app modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.model import extract_features

def main():
    print("=" * 60)
    print("GalaxEye ML Systems - Offline Model Training & Packaging")
    print("=" * 60)
    
    candidate_dir = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "candidate_tiles"
    eval_dir = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_set"
    eval_labels_path = settings.BASE_DIR / "be-mlsys-assignment-dataset" / "eval_labels.csv"
    
    if not candidate_dir.exists():
        print(f"Error: Candidate tiles directory not found at {candidate_dir}")
        return
        
    classes = sorted([d.name for d in candidate_dir.iterdir() if d.is_dir()])
    print(f"Identified {len(classes)} classes: {classes}")
    
    print("\n[1/3] Extracting spectral & spatial features from candidate tiles...")
    t0 = time.time()
    X_train = []
    y_train = []
    
    for c in classes:
        c_dir = candidate_dir / c
        images = list(c_dir.glob("*.png"))
        print(f"  - Loading class '{c}' ({len(images)} tiles)...")
        for img_p in images:
            img = Image.open(img_p)
            feat = extract_features(img)
            X_train.append(feat)
            y_train.append(c)
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Loaded {len(X_train)} training samples. Feature dimension: {X_train.shape[1]}")
    print(f"Extraction completed in {time.time() - t0:.2f}s")
    
    print("\n[2/3] Training offline Random Forest classifier...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - t0:.2f}s")
    
    # Evaluate against eval_set
    if eval_dir.exists() and eval_labels_path.exists():
        print("\n[3/3] Evaluating on independent evaluation set (eval_set)...")
        true_labels = {}
        with open(eval_labels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Header
            for row in reader:
                if row:
                    true_labels[row[0]] = row[1]
                    
        X_eval = []
        y_eval = []
        for fname in sorted(true_labels.keys()):
            img_p = eval_dir / fname
            if img_p.exists():
                img = Image.open(img_p)
                X_eval.append(extract_features(img))
                y_eval.append(true_labels[fname])
                
        X_eval = np.array(X_eval)
        y_eval = np.array(y_eval)
        
        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        print(f"\n==========================================")
        print(f" EVALUATION ACCURACY: {acc * 100:.2f}% ({sum(y_eval == y_pred)} / {len(y_eval)} correct)")
        print(f"==========================================")
        print("\nPer-class Classification Report:\n")
        print(classification_report(y_eval, y_pred, digits=3))
        
    # Serialize model bundle
    bundle = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_dim": X_train.shape[1],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "GalaxEye EuroSAT 7-Class Spatial-Spectral Classifier"
    }
    
    out_path = settings.MODEL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nModel artifact successfully saved to: {out_path} ({file_size_mb:.2f} MB)")
    print("The model is ready for 100% offline edge deployment!")

if __name__ == "__main__":
    main()
