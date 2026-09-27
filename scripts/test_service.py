import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

def test_pipeline():
    with TestClient(app) as client:
        # 1. Health check
        print("Testing GET /health...")
        resp = client.get("/health")
        print("Health response:", resp.status_code, resp.json())
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        
        # 2. Ingest a real tile from eval_set
        sample_tile = Path("be-mlsys-assignment-dataset/eval_set/tile_001.png")
        assert sample_tile.exists(), "Sample tile does not exist!"
        
        print("\nTesting POST /ingest with sample tile (tile_001.png)...")
        with open(sample_tile, "rb") as f:
            resp = client.post("/ingest", files={"file": ("tile_001.png", f, "image/png")})
        
        print("Ingest response:", resp.status_code, resp.json())
        assert resp.status_code == 201
        tile_data = resp.json()
        tile_id = tile_data["tile_id"]
        print(f"Ingested tile ID: {tile_id}, Predicted: {tile_data['predicted_class']}, Confidence: {tile_data['confidence']}")
        
        # 3. Ingest second tile
        sample_tile2 = Path("be-mlsys-assignment-dataset/eval_set/tile_035.png")
        with open(sample_tile2, "rb") as f:
            resp = client.post("/ingest", files={"file": ("tile_035.png", f, "image/png")})
        assert resp.status_code == 201
        print("Ingested tile 2:", resp.json()["predicted_class"], resp.json()["confidence"])

        # 4. Ingest via local path
        print("\nTesting POST /ingest-path...")
        sample_tile3 = Path("be-mlsys-assignment-dataset/eval_set/tile_070.png").resolve()
        resp = client.post("/ingest-path", json={"filepath": str(sample_tile3)})
        print("Ingest-path response:", resp.status_code, resp.json())
        assert resp.status_code == 201

        # 5. Query results
        print("\nTesting GET /query...")
        resp = client.get("/query?limit=10")
        print("Query response count:", resp.json()["count"])
        assert resp.status_code == 200
        assert resp.json()["count"] >= 3

        # 6. Query with class filter
        print("\nTesting GET /query?class_name=Forest...")
        resp = client.get("/query?class_name=Forest")
        print(f"Found {resp.json()['count']} forest tiles.")

        # 7. Query with review flag
        print("\nTesting GET /query?review_required=true...")
        resp = client.get("/query?review_required=true")
        print(f"Tiles requiring review: {resp.json()['count']}")

        # 8. Check stats
        print("\nTesting GET /stats...")
        resp = client.get("/stats")
        print("Stats response:", resp.json())
        assert resp.status_code == 200

        # 9. Test feedback
        print(f"\nTesting POST /tiles/{tile_id}/feedback...")
        resp = client.post(f"/tiles/{tile_id}/feedback", json={"verified_class": "Forest", "notes": "Verified by senior analyst"})
        print("Feedback response:", resp.status_code, resp.json())
        assert resp.status_code == 200

        print("\n==========================================")
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("==========================================")

if __name__ == "__main__":
    test_pipeline()
