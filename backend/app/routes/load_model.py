# routes/predict.py
from fastapi import APIRouter
from backend.app.models.model_loader import ModelLoader
from backend.app.services.logging_config import get_logger

logger = get_logger()
router = APIRouter()

@router.post("/load-latest-model")
async def load_latest_model():
    
    ModelLoader._model = None
    ModelLoader.get_model()
    
    return {
        "status": "success",
        "message": "YOLO model loaded and ready for inference",
        "version": ModelLoader._model_version
    }
