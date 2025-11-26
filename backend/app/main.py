# app/main.py
from fastapi import FastAPI
from backend.app.routes.predict import router as predict_router
from backend.app.routes.load_model import router as load_model_router
import uvicorn
from backend.app.config import API_HOST, API_PORT

# FastAPI app with root_path
app = FastAPI(
    title="FastAPI MLOps Backend",
    root_path="/api/v1",        # mounts app under /api/v1 for Ingress
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc UI
    openapi_url="/openapi.json" # OpenAPI JSON
)

# Include your routers
app.include_router(predict_router)
app.include_router(load_model_router)

# Optional: Health endpoint
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

# Run uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
