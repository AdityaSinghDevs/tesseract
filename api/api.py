from typing import Union
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse

from ..main import generate_from_prompt

app = FastAPI()

JOBS = {}

os.makedirs("../tesseract/outputs", exist_ok=True)

