
from backend.app.services.logging_config import get_logger
from dotenv import load_dotenv
import mlflow.pyfunc
import mlflow
import boto3
import os
import shutil
from ultralytics import YOLO


load_dotenv()
MLFLOW_URI = os.getenv("mlflow_uri")
ACCESS_KEY_ID = os.getenv("aws_access_key_id")
SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
REGIO_NAME = os.getenv("region_name")

logger = get_logger()

class ModelLoader:
    _model = None
    _model_name = "fire_vs_smoke_yolov8"
    _model_version = None

    @classmethod
    def _load_model(cls, tracking_uri: str = MLFLOW_URI):
        """
        Load and cache the latest Production model from MLflow.
        If already loaded, return the cached model.
        """

        boto3.setup_default_session(
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            region_name=REGIO_NAME
        )

        if cls._model is not None:
            return cls._model

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        # Get all versions for this model
        versions = client.search_model_versions(f"name='{cls._model_name}'")

        # Filter for Production stage
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if not prod_versions:
            raise ValueError(f"No Production versions found for model '{cls._model_name}'.")

        # Pick the latest by timestamp
        newest_prod = max(prod_versions, key=lambda v: v.last_updated_timestamp)

        cls._model_version = newest_prod.version
        logger.info(f"Loading Production model '{cls._model_name}' version {cls._model_version} ...")

        # Load model
        model_uri = f"models:/{cls._model_name}/{cls._model_version}"

        local_path = mlflow.artifacts.download_artifacts(model_uri)

        # Destination folder you want to save to
        dest_folder = "backend/model_metadata/"
        os.makedirs(dest_folder, exist_ok=True)

        # If the artifact is a file (like YOLO .pt)
        if os.path.isfile(local_path):
            dest_path = os.path.join(dest_folder, os.path.basename(local_path))
            shutil.copy(local_path, dest_path)
        else:
            # If it's a folder, copy the whole folder
            dest_path = os.path.join(dest_folder, os.path.basename(local_path))
            shutil.copytree(local_path, dest_path, dirs_exist_ok=True)
        
        for item in os.listdir(dest_path):
            path = os.path.join(dest_path, item)
            if os.path.isfile(path):
                os.remove(path)

        cls._model = YOLO(dest_path+"artifacts/best.pt")
        logger.info("Model loaded successfully.")
        return cls._model
    
    @classmethod
    def get_model(cls):
        """
        Returns the cached model. Loads it if not already loaded.
        """
        dest_file = "backend/model_metadata/artifacts/best.pt"
        if cls._model is None or os.path.exists(dest_file):
            logger.info("Model not loaded yet. Loading now...")
            cls._load_model()
        return cls._model
