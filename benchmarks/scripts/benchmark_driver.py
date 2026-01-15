"""
This script measures end-to-end and inference-only latency for a fixed prompt and configuration
"""
import torch

from typing import Any, Dict, List
from pathlib import Path

from benchmarks.configs.config_loader import load_benchmark_config
from benchmarks.utils.timing import measure_time, measure_peak_gpu_memory
from benchmarks.utils.io import load_prompt, write_raw_results

from tesseract.core.model_loader import get_device, load_all_models
from tesseract.core.generator import get_or_generate_latents
from tesseract.core.mesh_util import decode_latents, save_mesh

from tesseract.config.config import ( USE_CUDA,FALLBACK_TO_CPU,BASE_MODEL,
                                    TRANSMITTER,DIFFUSION_CONFIG, OUTPUT_DIR,
                                    BASE_FILE,
                                    GUIDANCE_SCALE, USE_FP16, USE_KARRAS, CLIP_DENOISED,PROGRESS,
                                    SIGMA_MIN, SIGMA_MAX, S_CHURN)

from tesseract.loggers.logger import get_logger

"""Benchmark Configs"""
BENCHMARK_NAME = "inference_latency"
CONFIG_VARIANT = "baseline"

DEVICE = "gpu" #CHANGE THIS FOR CPU RUN
BATCH_SIZE = 1 #Change this for batch size diff
NUM_RUNS = 3

PROMPT_PATH = Path("benchmarks/prompts/simple.txt")
PROMPT_NAME = PROMPT_PATH.stem

RESULTS_DIR = Path("benchmarks/results/raw/")

logger = get_logger(__name__ , log_file="benchmark_inf.log" )

logger.info("Starting inference latency benchmark")
logger.info(f"Config = {CONFIG_VARIANT}, Device = {DEVICE}, Batch Size = {BATCH_SIZE} , Runs = {NUM_RUNS}, Prompt Used: {PROMPT_NAME}")

logger.info(
    f"Run configuration | Device={DEVICE} | FP16={USE_FP16}"
)

"""Loading fixed inputs"""
resolved_config = load_benchmark_config(name = CONFIG_VARIANT, device=DEVICE, batch_size=BATCH_SIZE)

prompt = load_prompt(prompt_path=PROMPT_PATH)

"""Model Initialization"""

logger.info("Initializing device and loading models")

device = get_device(use_cuda=(DEVICE =='gpu'), fallback_to_cpu=True) # CHANGE THIS TO use_cuda=False FOR CPU RUNS

transmitter, base_model, diffusion_process = load_all_models(
    device=device,
    base_model=BASE_MODEL,
    transmitter=TRANSMITTER,
    diffusion_config=DIFFUSION_CONFIG,
)


logger.info("Model initialization complete")

"""Warm-Up run : Not measured"""

logger.info ("Performing warm-up run (discarded)")

_gen = get_or_generate_latents(
            prompt=prompt,
            model=base_model,
            diffusion=diffusion_process,
            base_file=BASE_FILE,
            output_dir=OUTPUT_DIR,
            resume=False,
            batch_size=BATCH_SIZE, guidance_scale=GUIDANCE_SCALE,
            progress = PROGRESS, clip_denoised=CLIP_DENOISED,
            use_fp16=USE_FP16,
            use_karras=USE_KARRAS,
            karras_steps=resolved_config["inference"]["sampling_steps"],
            sigma_max=SIGMA_MAX,
            sigma_min=SIGMA_MIN,
            s_churn=S_CHURN
        )
meshes = decode_latents(model=transmitter, latents=_gen)
_ = save_mesh(meshes=meshes, base_file="discard", output_dir=OUTPUT_DIR, formats=resolved_config["output"]["formats"])


logger.info("Warm-up completed")

"""BENCHMARK RESULTS"""


raw_results: List[Dict[str,Any]] = []

def run_inference()-> Any:

    latents = get_or_generate_latents(
            prompt=prompt,
            model=base_model,
            diffusion=diffusion_process,
            base_file=BASE_FILE,
            output_dir=OUTPUT_DIR,
            resume=False,
            batch_size=BATCH_SIZE, guidance_scale=GUIDANCE_SCALE,
            progress = PROGRESS, clip_denoised=CLIP_DENOISED,
            use_fp16=USE_FP16,
            use_karras=USE_KARRAS,
            karras_steps=resolved_config["inference"]["sampling_steps"],
            sigma_max=SIGMA_MAX,
            sigma_min=SIGMA_MIN,
            s_churn=S_CHURN
        )
    
    return latents

for run_index in range(NUM_RUNS):

    logger.info(f"Starting benchmark run {run_index + 1}/{NUM_RUNS}")

    # NOTE:
    # Inference-only latency is defined as *latent generation only*.
    # Post-processing and serialization are excluded from inference timing.

    if DEVICE != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    with measure_time() as end_to_end:
        
        with measure_time() as inference_time: 
            latents  = run_inference()
        inf = inference_time()
        
        meshes = decode_latents(model=transmitter, latents=latents)
        _ = save_mesh(meshes=meshes, base_file=f"run_{run_index}",
                              output_dir=OUTPUT_DIR, formats=resolved_config["output"]["formats"])


    e2e = end_to_end()

    if DEVICE != "cpu" and torch.cuda.is_available():
         mem = measure_peak_gpu_memory()
    else:
         mem = None

    run_record = {
        "benchmark":BENCHMARK_NAME,
        "run_index": run_index,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "prompt": PROMPT_NAME,
        "sampling_steps": resolved_config["inference"]["sampling_steps"],
        "end_to_end_seconds": e2e,
        "inference_seconds": inf,
        "peak_gpu_memory_bytes": mem
    }

    raw_results.append(run_record)

    logger.info(
        f"Run {run_index} completed | "
        f"E2E={e2e:.3f}s | "
        f"Inference={inf:.3f}s | "
        f"PeakMem={mem}"
    )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = RESULTS_DIR / f"{BENCHMARK_NAME}_{DEVICE}_{CONFIG_VARIANT}_batch_size_{BATCH_SIZE}_{PROMPT_NAME}.json"

write_raw_results(output_path=output_path, data=raw_results)
logger.info(f"Benchmark completed successfully")
logger.info(f"Raw results written to {output_path}")
         

