from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictRequest(BaseModel):
    markers: Dict[str, float] = Field(..., description="Feature name -> numeric value")

class PredictResponse(BaseModel):
    diagnosis: str
    confidence: Optional[float] = Field(None, description="Probability if available")
