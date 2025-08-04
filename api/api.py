from typing import Union
import os
import uuid

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse

from ..main import generate_from_prompt

app = FastAPI()

JOBS = {}

os.makedirs("../tesseract/outputs", exist_ok=True)

@app.post("/generate")
def generate_endpoint(prompt:str, fmt:str = 'ply', background_tasks : BackgroundTasks = None):

    job_id = str(uuid.uuid4())
    JOBS[job_id] ={"status" : "queued" ,
                   "file": None}
    
    def run_generation():
        JOBS[job_id]["status"] = "running"
        file_path = generate_from_prompt(prompt=prompt ,
                                         formats=fmt)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["file"] = "saved_files"

    background_tasks.add_task(run_generation)

    return {"job_id" : job_id, "status" : "queued"}