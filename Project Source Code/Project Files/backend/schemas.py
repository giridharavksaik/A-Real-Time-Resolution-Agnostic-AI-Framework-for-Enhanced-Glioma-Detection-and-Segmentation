from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    mask_b64: str
    heatmap_b64: str
    summary: str

class MetricsResponse(BaseModel):
    accuracy: float
    loss: float
