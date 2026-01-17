from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from numbers import Integral

import joblib


@dataclass
class SklearnModelService:
    model_path: Path

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def predict_with_feature_order(self, markers: Dict[str, float], feature_order: list[str]) -> Tuple[str, float]:
        x = [[float(markers[f]) for f in feature_order]]
        pred = self.model.predict(x)[0]

        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]

        if isinstance(pred, Integral):
            diagnosis = "Malignant" if int(pred) == 1 else "Benign"
        else:
            s = str(pred).strip()
            if s == "1":
                diagnosis = "Malignant"
            elif s == "0":
                diagnosis = "Benign"
            else:
                diagnosis = s

        confidence = 0.5
        if proba is not None:
            confidence = float(max(proba))

        return diagnosis, confidence
