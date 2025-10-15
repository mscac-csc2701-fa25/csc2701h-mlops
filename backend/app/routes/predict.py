# routes/predict.py
from fastapi import APIRouter, UploadFile, File, Depends
from app.models.model_loader import ModelLoader
from app.services.inference_service import run_inference
from app.services.logging_config import get_logger

logger = get_logger()

router = APIRouter()
model = ModelLoader()

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    logger.info(f"Loading images ...")
    pred = run_inference(model, image_bytes)
    return {"filename": file.filename, "prediction": str(pred)}
