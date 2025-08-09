from typing import List, Any 
import torch

from ..config.config import USE_CUDA, FALLBACK_TO_CPU, BASE_MODEL, TRANSMITTER, DIFFUSION_CONFIG
from ..loggers.logger import get_logger
from .shap_e.models.download import load_model, load_config
from .shap_e.diffusion.gaussian_diffusion import diffusion_from_config

logger = get_logger(__name__ , log_file= "app.log")

def get_device(use_cuda : bool=USE_CUDA,
fallback_to_cpu : bool=  FALLBACK_TO_CPU)->torch.device :
     '''
     Selects the computation device for model execution.

    Prefers CUDA if available and enabled; otherwise falls back to CPU 
    if configured. Raises an error if neither option is available.

    Args:
        use_cuda (bool): Attempt to use CUDA if available.
        fallback_to_cpu (bool): Use CPU if CUDA is unavailable.

    Returns:
        torch.device: Selected computation device.

    Raises:
        RuntimeError: If CUDA is unavailable and CPU fallback is disabled.
     '''
     
     
     if use_cuda and torch.cuda.is_available():
          device_name = torch.cuda.get_device_name(0)
          logger.info(f"CUDA found, device : {device_name}")
          return torch.device('cuda')
     elif fallback_to_cpu:
          logger.warning("CUDA not found, falling back to CPU")
          return torch.device('cpu')
     else:
          logger.error("CUDA not found, CPU disabled")
          raise RuntimeError("CUDA is not available and CPU is disabled in configs")
     

def load_all_models(device : torch.device, base_model :str = BASE_MODEL, transmitter: str = TRANSMITTER,
diffusion_config : str = DIFFUSION_CONFIG)-> List[Any]:
     
     '''
     Loads all required models and initializes the diffusion process.

    Args:
        device (torch.device): Target device for model loading.
        base_model (str): Name of the base text encoder model.
        transmitter (str): Name of the transmitter model.
        diffusion_config (str): Path to diffusion process configuration.

    Returns:
        List[Any]: [transmitter_model, text_encoder_model, diffusion_process].

    Raises:
        TypeError: If `device` is not a torch.device instance.
    '''
     
     if not isinstance(device, torch.device):
          logger.error(f"Invalid device type : {type(device)}")
          raise TypeError("Device must be a torch.device object")
     
     transmitter_model = load_model(model_name = transmitter, device=device)
     logger.info(f"Transmitter model '{transmitter}' loaded on {device.type}")

     text_encoder_model = load_model(model_name = base_model, device=device )
     logger.info(f"Base model loaded : {base_model} loaded on {device.type}")
     
     diffusion_process = diffusion_from_config(load_config(diffusion_config))
     logger.info(f"Diffusion process intitiated...")

     return [transmitter_model, text_encoder_model, diffusion_process ]
