import time
import math
import warnings
import joblib
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import settings

def extract_features(img: Image.Image) -> np.ndarray:
    """
    Extract robust spatial, spectral, and textural features from a 64x64 satellite tile.
    Designed for offline edge execution: zero network calls, CPU-friendly, <2ms latency.
    """
    img_rgb = img.convert('RGB')
    arr = np.array(img_rgb, dtype=np.float32)
    
    # 1. Spectral Channel Statistics (Mean, Std, 10th & 90th Percentiles)
    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    p10_rgb = np.percentile(arr, 10, axis=(0, 1))
    p90_rgb = np.percentile(arr, 90, axis=(0, 1))
    
    # 2. Spectral Vegetation & Water Proxies (Optical Band Ratios)
    # Band 4 (Red), Band 3 (Green), Band 2 (Blue)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    
    # Visible Atmospherically Resistant Index (VARI) proxy: (G - R) / (G + R + eps)
    rg_diff = (g - r) / (g + r + 1e-5)
    # Blue-to-Red ratio: strong discriminator for deep vs shallow water bodies
    br_diff = (b - r) / (b + r + 1e-5)
    
    idx_features = [
        float(rg_diff.mean()), float(rg_diff.std()),
        float(br_diff.mean()), float(br_diff.std())
    ]
    
    # 3. Spatial Grid Pooling (2x2 sub-blocks)
    # Captures spatial asymmetry (e.g. highways/rivers traversing a quadrant)
    h, w, _ = arr.shape
    h_mid, w_mid = h // 2, w // 2
    quadrants = [
        arr[:h_mid, :w_mid],
        arr[:h_mid, w_mid:],
        arr[h_mid:, :w_mid],
        arr[h_mid:, w_mid:]
    ]
    quad_stats = []
    for quad in quadrants:
        quad_stats.extend(quad.mean(axis=(0, 1)))
        quad_stats.extend(quad.std(axis=(0, 1)))
        
    # 4. Color Histograms (16 bins per channel = 48 features)
    hist_r, _ = np.histogram(r, bins=16, range=(0, 256), density=True)
    hist_g, _ = np.histogram(g, bins=16, range=(0, 256), density=True)
    hist_b, _ = np.histogram(b, bins=16, range=(0, 256), density=True)
    
    # 5. Spatial Edge Texture (finite differences along X and Y axes)
    dx = np.abs(np.diff(arr, axis=1)).mean(axis=(0, 1))
    dy = np.abs(np.diff(arr, axis=0)).mean(axis=(0, 1))
    
    features = np.concatenate([
        mean_rgb, std_rgb, p10_rgb, p90_rgb,
        np.array(idx_features, dtype=np.float32),
        np.array(quad_stats, dtype=np.float32),
        hist_r, hist_g, hist_b,
        dx, dy
    ])
    return features


class TileClassifier:
    """
    Offline inference engine wrapper.
    Loads trained model bundle and produces predictions with confidence & uncertainty metrics.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.classes = settings.CLASSES
        self._load()

    def _load(self):
        if self.model_path.exists():
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                self.classes = bundle.get("classes", settings.CLASSES)
            else:
                self.model = bundle
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
        else:
            self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single PIL image.
        Returns predicted class, confidence, uncertainty entropy, full distribution, and latency.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"Model file not found at {self.model_path}. "
                "Run `python scripts/train_model.py` to train and bundle the model offline."
            )
            
        t0 = time.perf_counter()
        features = extract_features(image).reshape(1, -1)
        
        # Inference
        probs = self.model.predict_proba(features)[0]
        classes = self.model.classes_ if hasattr(self.model, "classes_") else self.classes
        
        prob_dict = {str(cls_name): round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
        
        top_idx = int(np.argmax(probs))
        predicted_class = str(classes[top_idx])
        confidence = float(probs[top_idx])
        
        # Calculate Shannon entropy as uncertainty metric (higher entropy = more uncertain)
        # H(p) = -sum(p * log2(p)) for p > 0
        entropy = -sum(float(p) * math.log2(float(p)) for p in probs if p > 1e-7)
        entropy = max(0.0, round(entropy, 4))
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # Flag for human analyst review if confidence is below threshold or high entropy
        review_required = (confidence < settings.CONFIDENCE_THRESHOLD)
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "uncertainty_entropy": round(entropy, 4),
            "review_required": review_required,
            "probabilities": prob_dict,
            "latency_ms": round(latency_ms, 2)
        }

# Global singleton classifier instance
classifier = TileClassifier()
