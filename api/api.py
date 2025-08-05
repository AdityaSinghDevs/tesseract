from typing import Union, Dict
import os
import uuid

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from ..main import generate_from_prompt, initialize_pipeline, BASE_FILE, OUTPUT_DIR
from ..tesseract.loggers.logger import get_logger

logger = get_logger(__name__, log_file='api.log')

app = FastAPI()

logger.info("Starting FastApi server...")

PIPELINE = initialize_pipeline()

os.makedirs(OUTPUT_DIR, exist_ok=True) #making surre ouput dir exists

@app.get("/")
def root():
    return ("Hello this is the root route for this api endpoint")

@app.post("/generate")
def 



















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
