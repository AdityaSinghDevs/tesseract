import os 
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "defaults.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)
cfg = load_config()


PROJECT_NAME = cfg["general"]["project_name"]
TASK = cfg["general"]["task"]
MODEL_NAME = cfg["general"]["model"]
SEED = cfg["general"]["seed"]


USE_CUDA = cfg["device"]["use_cuda"]
FALLBACK_TO_CPU = cfg["device"]["fallback_to_cpu"]


OUTPUT_DIR = cfg["paths"]["output_dir"]
RENDER_DIR = cfg["paths"]["render_dir"]
MODEL_CACHE_DIR = cfg["paths"]["model_cache_dir"]


SAVE_RENDER = cfg["generation"]["save_render"]
RENDER_RESOLUTION = tuple(cfg["generation"]["render_resolution"])
