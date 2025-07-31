import os 
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "defaults.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)
cfg = load_config()

#General
PROJECT_NAME = cfg["general"]["project_name"]
MODEL_NAME = cfg["general"]["model"]

#models
BASE_MODEL = cfg["general"]["base_model"]
TRANSMITTER = cfg["general"]["transmitter"]


USE_FP16 = cfg["general"]["use_fp16"]
BATCH_SIZE = cfg["general"]["batch_size"]

#Device
USE_CUDA = cfg["device"]["use_cuda"]
FALLBACK_TO_CPU = cfg["device"]["fallback_to_cpu"]

#diffusion
DIFFUSION_CONFIG = cfg["diffusion"]["config_type"]

#latents 
LATENT_BATCH_SIZE = cfg["latents"]["batch_size"]
GUIDANCE_SCALE = cfg["latents"]["guidance_scale"]
USE_FP16 = cfg["latents"]["use_fp16"]
USE_KARRAS = cfg["latents"]["use_karras"]
KARRAS_STEPS = cfg["latents"]["karras_steps"]
CLIP_DENOISED = cfg["latents"]["clip_denoised"]
PROGRESS = cfg["latents"]["progress"]
SIGMA_MIN = cfg["latents"]["sigma_min"]
SIGMA_MAX = cfg["latents"]["sigma_max"]
S_CHURN = cfg["latents"]["s_churn"]

#filesaving
OUTPUT_DIR = cfg["file"]["output_dir"]
DEFAULT_FORMATS = cfg["file"]["default_format"]
BASE_FILE = cfg['file']['base_file']