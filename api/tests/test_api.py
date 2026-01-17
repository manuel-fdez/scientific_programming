from fastapi.testclient import TestClient
from api.app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_schema_features():
    r = client.get("/schema/features")
    assert r.status_code == 200
    feats = r.json()["features"]
    assert isinstance(feats, list)
    assert len(feats) == 20

def test_predict_success():
    feats = client.get("/schema/features").json()["features"]
    payload = {"markers": {f: 0.0 for f in feats}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["diagnosis"] in ["Benign", "Malignant"]
    assert 0.0 <= float(out["confidence"]) <= 1.0

def test_predict_missing_feature_422():
    feats = client.get("/schema/features").json()["features"]
    payload = {"markers": {f: 0.0 for f in feats}}
    payload["markers"].pop(feats[0])
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
