import uvicorn
from fastapi import FastAPI

from .api.api import router, lifespan
from .tesseract.loggers.logger import get_logger

logger = get_logger(__name__, log_file="server.log")


app = FastAPI(
    title="TesseractV1 API",
    version="1.0.0",
    description="Text-to-3D mesh generation using Shap-E",
    lifespan=lifespan
)


app.include_router(router)

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "message": "TesseractV1 API is running!"}


if __name__ == "__main__":
    logger.info("Starting FastAPI server locally...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)