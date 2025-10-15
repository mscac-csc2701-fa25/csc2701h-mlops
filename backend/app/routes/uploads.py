# routes/uploads.py
from fastapi import APIRouter, UploadFile, File
import shutil
import os
from app.services.logging_config import get_logger

router = APIRouter()
logger = get_logger()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"Saving uploaded file to {file_path}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "file_path": file_path}
