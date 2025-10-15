# app/main.py
from fastapi import FastAPI
from app.routes.uploads import router as uploads_router
from app.routes.predict import router as predict_router
import uvicorn
from app.config import API_HOST, API_PORT

# FastAPI app with root_path
app = FastAPI(
    title="FastAPI MLOps Backend",
    root_path="/api/v1",        # mounts app under /api/v1 for Ingress
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc UI
    openapi_url="/openapi.json" # OpenAPI JSON
)

# Include your routers
app.include_router(uploads_router)
app.include_router(predict_router)

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
