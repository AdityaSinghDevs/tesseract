"""
Simple model loader for SHAP-E production pipeline
No classes, just functions that work reliably
"""

import torch
from loggers.logger import get_logger
from typing import Dict, Any, Optional
from config import MODEL_CACHE_DIR, USE_FP16, GUIDANCE_SCALE, BATCH_SIZE
from .shap_e_core.download import load_model, load_config
from .shap_e_core.gaussian_diffusion import diffusion_from_config

logger = get_logger(); #TODO: add parameter

# Global variables to cache models (simple approach)
_device = None
_models_cache = {}
_diffusion = None

def check_and_set_device() -> torch.device:
    """
    Check CUDA availability and set device
    Returns the device to use
    """
    global _device
    
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available - Using GPU: {gpu_name} ({memory_gb:.1f}GB)")
            
            # Clear any existing cache
            torch.cuda.empty_cache()
        else:
            _device = torch.device('cpu')
            logger.warning("CUDA not available - Using CPU (will be slower)")
    
    return _device

# def get_tokenizer():
#     """
#     SHAP-E doesn't use a traditional tokenizer like text models
#     It processes text directly through the text encoder
#     This function is here for API consistency but returns None
#     """
#     logger.debug("SHAP-E uses built-in text encoding, no separate tokenizer needed")
#     return None

def load_shap_e_models(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load SHAP-E models (transmitter + text model + diffusion)
    Only loads once unless force_reload=True
    
    Args:
        force_reload: Force reload models even if cached
        
    Returns:
        Dict containing loaded models and diffusion
    """
    global _models_cache, _diffusion
    
    device = check_and_set_device()
    
    # Return cached models if available and not forcing reload
    if not force_reload and _models_cache and _diffusion:
        logger.debug("Using cached models")
        return {
            'transmitter': _models_cache.get('transmitter'),
            'text_model': _models_cache.get('text_model'),
            'diffusion': _diffusion,
            'device': device
        }
    
    try:
        logger.info("Loading SHAP-E models...")
        
        # Load transmitter (converts latents to 3D)
        logger.info("Loading transmitter model...")
        transmitter = load_model('transmitter', device=device)
        if USE_FP16 and device.type == 'cuda':
            transmitter = transmitter.half()
        _models_cache['transmitter'] = transmitter
        
        # Load text-conditional model
        logger.info("Loading text model...")
        text_model = load_model('text300M', device=device)
        if USE_FP16 and device.type == 'cuda':
            text_model = text_model.half()
        _models_cache['text_model'] = text_model
        
        # Load diffusion config and create diffusion model
        logger.info("Loading diffusion configuration...")
        diffusion_config = load_config('diffusion')
        _diffusion = diffusion_from_config(diffusion_config)
        
        logger.info("All SHAP-E models loaded successfully")
        
        return {
            'transmitter': transmitter,
            'text_model': text_model, 
            'diffusion': _diffusion,
            'device': device
        }
        
    except Exception as e:
        logger.error(f"Failed to load SHAP-E models: {e}")
        # Clear cache on failure
        clear_model_cache()
        raise

def get_loaded_models() -> Optional[Dict[str, Any]]:
    """
    Get currently loaded models without reloading
    Returns None if models aren't loaded yet
    """
    if _models_cache and _diffusion:
        return {
            'transmitter': _models_cache.get('transmitter'),
            'text_model': _models_cache.get('text_model'),
            'diffusion': _diffusion,
            'device': _device
        }
    return None

def check_models_loaded() -> bool:
    """
    Check if models are already loaded in memory
    """
    return bool(_models_cache and _diffusion)

def get_memory_info() -> Dict[str, Any]:
    """
    Get current memory usage information
    """
    info = {
        'device': str(_device) if _device else 'not_set',
        'models_loaded': list(_models_cache.keys()) if _models_cache else [],
        'diffusion_loaded': _diffusion is not None
    }
    
    if _device and _device.type == 'cuda':
        try:
            info.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_cached_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
            })
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")
    
    return info

def clear_gpu_cache():
    """
    Clear GPU memory cache
    """
    if _device and _device.type == 'cuda':
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared")

def clear_model_cache():
    """
    Clear model cache and free memory
    Use this when you want to free up memory
    """
    global _models_cache, _diffusion
    
    _models_cache.clear()
    _diffusion = None
    clear_gpu_cache()
    logger.info("Model cache cleared")

def validate_generation_params(guidance_scale: float, batch_size: int) -> Dict[str, Any]:
    """
    Validate and return cleaned generation parameters
    
    Args:
        guidance_scale: Guidance scale for generation (higher = more faithful to prompt)
        batch_size: Number of samples to generate
        
    Returns:
        Dict of validated parameters
    """
    # Clamp guidance scale to reasonable range
    guidance_scale = max(1.0, min(guidance_scale, 50.0))
    
    # Clamp batch size to reasonable range  
    batch_size = max(1, min(batch_size, 8))
    
    # Memory check for large batches
    if batch_size > 2 and _device and _device.type == 'cuda':
        try:
            available_memory = torch.cuda.get_device_properties(0).total_memory
            if available_memory < 8 * 1024**3:  # Less than 8GB
                logger.warning("Low GPU memory detected, reducing batch size to 1")
                batch_size = 1
        except:
            pass
    
    params = {
        'guidance_scale': guidance_scale,
        'batch_size': batch_size,
        'use_fp16': USE_FP16 and _device and _device.type == 'cuda',
        'use_karras': True,  # Generally better quality
        'karras_steps': 64,   # Good balance of quality/speed
        'clip_denoised': True
    }
    
    logger.debug(f"Validated generation params: {params}")
    return params

def health_check() -> Dict[str, Any]:
    """
    Perform a health check of the model loading system
    Returns status information
    """
    status = {
        'device_available': False,
        'models_loaded': False,
        'memory_ok': True,
        'errors': []
    }
    
    try:
        # Check device
        device = check_and_set_device()
        status['device_available'] = True
        status['device'] = str(device)
        
        # Check if models are loaded
        status['models_loaded'] = check_models_loaded()
        
        # Check memory if using GPU
        if device.type == 'cuda':
            try:
                memory_info = get_memory_info()
                allocated = memory_info.get('gpu_memory_allocated_gb', 0)
                if allocated > 10:  # More than 10GB might be concerning
                    status['memory_ok'] = False
                    status['errors'].append(f"High GPU memory usage: {allocated:.1f}GB")
            except Exception as e:
                status['errors'].append(f"Could not check GPU memory: {e}")
        
    except Exception as e:
        status['errors'].append(f"Health check failed: {e}")
    
    return status

def prepare_for_generation() -> Dict[str, Any]:
    """
    Ensure everything is ready for generation
    This is the main function to call before generating
    
    Returns:
        Loaded models and status info
    """
    try:
        # Ensure device is set
        device = check_and_set_device()
        
        # Load models if not already loaded
        models = load_shap_e_models()
        
        # Quick memory cleanup
        clear_gpu_cache()
        
        logger.info("Ready for generation")
        return models
        
    except Exception as e:
        logger.error(f"Failed to prepare for generation: {e}")
        raise