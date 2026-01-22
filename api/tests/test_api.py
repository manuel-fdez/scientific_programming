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

def test_predict_label_mapping_matches_model_output():
    import pandas as pd
    import joblib

    df = pd.read_csv("data/breast_cancer_reduced.csv")
    X = df.drop(columns=["diagnosis", "diagnosis_label"])

    model = joblib.load("api/artifacts/best_model.pkl")

    idx_pred_0 = None
    idx_pred_1 = None

    # Find one example the model predicts 0 and one example it predicts 1
    for i in range(len(X)):
        row = X.iloc[i]
        x = [[float(row[c]) for c in X.columns]]
        pred = int(model.predict(x)[0])

        if pred == 0 and idx_pred_0 is None:
            idx_pred_0 = i
        if pred == 1 and idx_pred_1 is None:
            idx_pred_1 = i

        if idx_pred_0 is not None and idx_pred_1 is not None:
            break

    assert idx_pred_0 is not None, "Could not find any row predicted as 0 by the model"
    assert idx_pred_1 is not None, "Could not find any row predicted as 1 by the model"

    def payload_for_index(i: int):
        row = X.iloc[i].to_dict()
        return {"markers": {k: float(v) for k, v in row.items()}}

    # If model predicts 0 -> API should map to Malignant
    r0 = client.post("/predict", json=payload_for_index(idx_pred_0))
    assert r0.status_code == 200
    assert r0.json()["diagnosis"] == "Malignant"

    # If model predicts 1 -> API should map to Benign
    r1 = client.post("/predict", json=payload_for_index(idx_pred_1))
    assert r1.status_code == 200
    assert r1.json()["diagnosis"] == "Benign"

