from typing import Dict
import os
import zipfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, APIRouter, HTTPException
from fastapi.responses import FileResponse

from schemas import GenerateRequests, GenerateResponse, ErrorResponse
from ..main import generate_from_prompt, initialize_pipeline, BASE_FILE, OUTPUT_DIR
from ..tesseract.loggers.logger import get_logger

logger = get_logger(__name__, log_file='api.log')

router = APIRouter(prefix ="/api/v1", tags=[])

PIPELINE = None
JOBS: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app):
    global PIPELINE
    try:
        logger.info("Startinng up FastAPI app and initializing pipeline...")
        PIPELINE = initialize_pipeline()
        logger.info("Pipeline initiated successfully.")
        yield
    finally:
        logger.info("Shutting down FastAPI app. Cleanup if needed.")

def process_generation_job(job_id: str, request: GenerateRequests):
    JOBS[job_id]["status"] = "running"

    try:
        logger.info(f"JOb {job_id} started: prompt = '{request.prompt}'")

        result = generate_from_prompt(
            prompt=request.prompt,
            base_file=request.base_file,
            output_dir="tesseract/tesseract/outputs",
            formats = request.formats,
            preloaded_pipeline=PIPELINE,
            resume_latents = request.resume_latents
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
        



















JOBS : Dict[str, Dict] = {} #in memory job store


@app.post("/generate")
def generate_endpoint(prompt:str, fmt:str = 'ply', background_tasks : BackgroundTasks = None):

    job_id = str(uuid.uuid4()) #creatin a random id
    JOBS[job_id] ={"status" : "queued" , #initializing its  initial values
                   "files": []}
    
    def run_generation(): #fn made to run
        JOBS[job_id]["status"] = "running" #status updated
        result = generate_from_prompt(prompt=prompt ,
                                         formats= fmt,  base_file=BASE_FILE,
                                         output_dir=OUTPUT_DIR,
                                         )
        
        JOBS[job_id]["status"] = "done" #result info saved
        JOBS[job_id]["files"] = result["saved_files"]

    background_tasks.add_task(run_generation)

    return {"job_id" : job_id, "status" : "queued"}

@app.get("/status/{job_id}")
def check_status(job_id : str):
    job = JOBS.get(job_id)
    if not job:
        return {"error" : "job not found" }
    return job


@app.get("/download/{job_id}")
def download_file(job_id : str):
    job = JOBS.get(job_id)
    if not job :
        return {"error" : "job not found" }
    if job["status"] != "done":
        return {"status":job["status"], "message" : "File not ready yet"}
    
    file_path = job["files"][0]
    return FileResponse(file_path, filename = os.path.basename(file_path))

# @app.get("/render")
# def render_endpoint():
#     images = decode_latent_images(...)
#     widget = gif_widget(images)
#     return HTMLResponse(widget._repr_html_())