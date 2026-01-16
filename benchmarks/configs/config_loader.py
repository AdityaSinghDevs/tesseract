import yaml 
from pathlib import Path
from typing import Dict, Union

CONFIG_DIR = Path(__file__).parent

def load_benchmark_config( name: str , device: str, batch_size: int)-> Dict:
    if name not in {"baseline", "high_cost"}:
        raise ValueError(f"Unknown benchmark config: {name}")

    if device not in {"cpu" ,"gpu"}:
        raise ValueError(f"Invalid device : {device}")
    
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    config_path = CONFIG_DIR / f"{name}.yaml"

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    raw_cfg["runtime"]["device"] = device
    raw_cfg["inference"]["batch_size"] = batch_size

    return raw_cfg