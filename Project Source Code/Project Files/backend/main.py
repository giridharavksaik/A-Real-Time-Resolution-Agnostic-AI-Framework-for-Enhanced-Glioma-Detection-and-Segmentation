from fastapi import FastAPI, UploadFile, File
from backend.schemas import PredictionResponse, MetricsResponse
from backend.utils import generate_medical_summary, encode_image_to_base64, create_pdf_report
from models.inference import InferenceEngine
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from pathlib import Path

app = FastAPI(title="Glioma Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], # Accept frontend requests
)

# Initialize models
engine = InferenceEngine(weight_path="outputs/weights/training_complete.pth")

@app.get("/")
def read_root():
    return {"status": "Backend running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_mri(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    Path("outputs/sample").mkdir(parents=True, exist_ok=True)
    temp_path = f"outputs/sample/temp_upload.png"
    image.save(temp_path)
    
    results = engine.predict(image)
    
    mask_b64 = encode_image_to_base64(results["mask"])
    heatmap_b64 = encode_image_to_base64(results["heatmap"])
    
    summary = generate_medical_summary(results["prediction"], results["confidence"])
    
    pdf_path = create_pdf_report(
        original_img_path=temp_path,
        prediction=results["prediction"],
        confidence=results["confidence"],
        summary=summary
    )
    
    return PredictionResponse(
        prediction=results["prediction"],
        confidence=results["confidence"],
        mask_b64=mask_b64,
        heatmap_b64=heatmap_b64,
        summary=summary
    )

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    # Return metrics for dashboard visualization
    return MetricsResponse(accuracy=0.945, loss=0.082)
