import uvicorn
from fastapi import FastAPI

from api.api import router, lifespan
from tesseract.loggers.logger import get_logger

logger = get_logger(__name__, log_file="server.log")


app = FastAPI(
    title="TesseractV1 API",
    version="1.0.0",
    description="Text-to-3D mesh generation using Shap-E",
    lifespan=lifespan
)

'''
FastAPI application instance for the TesseractV1 service.

Configures metadata and application lifecycle for the text-to-3D mesh generation API.
'''

app.include_router(router)
'''
Includes all api routes for application in router
'''

@app.get("/", tags=["health"])
async def root():
    '''
    Health check endpoint.

    Returns:
        dict: API status and readiness message.
    '''
    return {"status": "ok", "message": "TesseractV1 API is running!"}


if __name__ == "__main__":
    '''
    Entry point for local development server.

    Launches FastAPI via Uvicorn with reload enabled for hot reloading.
    '''
    logger.info("Starting FastAPI server locally...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)