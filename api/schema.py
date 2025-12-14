from pydantic import BaseModel
from typing import Dict

class HealthResponse(BaseModel):
    status: str
    message: str

class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    latency_ms: float