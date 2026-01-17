import json
from pathlib import Path
from fastapi import FastAPI, HTTPException

from api.app.schemas import PredictRequest, PredictResponse
from api.app.model import SklearnModelService

app = FastAPI(
    title="Scientific Programming - Model API",
    version="0.4.0",
    description="FastAPI service for Collaborator 6 (deployment)."
)

BASE_DIR = Path(__file__).resolve().parents[2]  # repo_root/
ARTIFACTS_DIR = BASE_DIR / "api" / "artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"

model_service = SklearnModelService(model_path=MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema/features")
def schema_features():
    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=500, detail="features.json not found in api/artifacts/")
    features = json.loads(FEATURES_PATH.read_text())
    return {"features": features}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=500, detail="features.json not found in api/artifacts/")
    features = json.loads(FEATURES_PATH.read_text())

    missing = [f for f in features if f not in req.markers]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    diagnosis, confidence = model_service.predict_with_feature_order(req.markers, features)
    return PredictResponse(diagnosis=diagnosis, confidence=confidence)
