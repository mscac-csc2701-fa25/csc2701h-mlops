# routes/predict.py
from fastapi import APIRouter, UploadFile, File, Depends
from backend.app.models.model_loader import ModelLoader
from backend.app.services.inference_service import run_inference
from backend.app.services.logging_config import get_logger
from fastapi.responses import Response
import shutil
import os

logger = get_logger()
UPLOAD_DIR = "backend/uploads"
router = APIRouter()

@router.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"Saving uploaded file to {file_path}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    shutil.rmtree("backend/runs", ignore_errors=True)

    logger.info(f"Loading images ...")
    try:
        model = ModelLoader.get_model()
    except Exception as e:
        logger.exception(f"Failed to load model: {e}")

    logger.info(f"Running inference ...")
    
    run_inference(model, "backend/runs/", "backend/uploads/")
    shutil.rmtree("backend/uploads")
    with open("backend/runs/output.jpg", "rb") as f:
        img_bytes = f.read()
    shutil.rmtree("backend/runs", ignore_errors=True)
    return Response(content=img_bytes, media_type="image/jpeg")
