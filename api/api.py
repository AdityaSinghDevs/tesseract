from typing import Dict
import os
import zipfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.schemas import GenerateRequests, GenerateResponse
from main import generate_from_prompt, initialize_pipeline, BASE_FILE, OUTPUT_DIR
from tesseract.loggers.logger import get_logger

logger = get_logger(__name__, log_file='api.log')

router = APIRouter(prefix ="/api/v1", tags=[])

PIPELINE = None
JOBS: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app):
    '''
    Initialize and manage the FastAPI application lifecycle.

    Loads the global pipeline at startup and performs cleanup on shutdown.
    '''
    global PIPELINE
    try:
        logger.info("Startinng up FastAPI app and initializing pipeline...")
        PIPELINE = initialize_pipeline()
        logger.info("Pipeline initiated successfully.")
        yield
    finally:
        logger.info("Shutting down FastAPI app. Cleanup if needed.")

def process_generation_job(job_id: str, request: GenerateRequests):
    '''
    Execute a generation job for a given prompt and store results.

    Updates the global JOBS registry with status, results, or errors.
    '''
    JOBS[job_id]["status"] = "running"

    try:
        logger.info(f"JOb {job_id} started: prompt = '{request.prompt}'")

        result = generate_from_prompt(
            prompt=request.prompt,
            base_file=request.base_file,
            guidance_scale=request.guidance_scale,
            karras_steps=request.karras_steps,
            output_dir="tesseract/api_outputs",
            formats = request.formats,
            preloaded_pipeline=PIPELINE,
            resume_latents = request.resume_latents,
            batch_size=request.batch_size
            
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = GenerateResponse(
            status="success",
            prompt=result["prompt"],
            mesh_count=result["mesh_count"],
            saved_files=result["saved_files"],
            latents_path=result.get("latents_path"),
            output_dir=result.get("output_dir"),
            job_id=job_id,
        ).model_dump()

        logger.info(f"JOb {job_id} completed ({result['mesh_count']} meshes)")

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        logger.error(f" Job {job_id} failed: {e}", exc_info=True)


@router.post("/generate")
async def generate_endpoint(request: GenerateRequests,
                            background_tasks : BackgroundTasks):
    '''
    Queue a generation job for asynchronous processing.

    Returns a job ID for status polling via the /status endpoint.
    '''
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "result":None, "error":None}

    background_tasks.add_task(process_generation_job, job_id, request)
    logger.info(f"Job {job_id} queued for prompt '{request.prompt}'")

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Job queued successfully. Poll /api/v1/status/{job_id} for updates."
    }

@router.get("/status/{job_id}")
async def check_status(job_id: str):
    '''
    Retrieve the current status of a generation job.

    Returns job state and progress; raises 404 if job is unknown.
    '''
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": job_id, "status": job["status"]}


@router.get("/download/{job_id}")
async def download_files(job_id:str):
    '''
     Package and return generated mesh files as a ZIP archive.

    Raises an error if the job is incomplete or no files are available.
    '''
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job["status"] != "completed" or not job["result"]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    output_dir = job["result"]["output_dir"]
    saved_files = job["result"]["saved_files"]

    if not saved_files:
       raise HTTPException(status_code=404, detail="No files available to download")  
    
    zip_path = os.path.join(output_dir, f"{job_id}_meshes.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in saved_files:
            if os.path.exists(file):
                zipf.write(file, arcname=os.path.basename(file))

    return FileResponse(zip_path, filename=f"{job_id}_meshes.zip", media_type = "application/zip")


