from dotenv import load_dotenv
import os

load_dotenv()

MLFLOW_SERVER = os.getenv("MLFLOW_SERVER")
DATA_YAML = "data/processed/data.yaml"
