from typing import Any

from ..loggers.logger import get_logger
from ..config.config import (
    LATENT_BATCH_SIZE,
    GUIDANCE_SCALE,
    USE_FP16,
    USE_KARRAS,
    KARRAS_STEPS,
    CLIP_DENOISED,
    PROGRESS,
    SIGMA_MIN,
    SIGMA_MAX,
    S_CHURN,
)
from shap_e.diffusion.sample import sample_latents

logger = get_logger(__name__, log_file='app.log')

def validate_inputs(prompt : str, model : Any ,
                     diffusion : Any)->None :
    if not isinstance(prompt,str) or not prompt.strip():
        logger.error("Empty or Invalid Prompt given")
        raise ValueError("The Prompt should be a str and not empty")
    if not model:
        logger.error("Model not provided")
        raise ValueError("Model must be provided and not none")
    if not diffusion:
        logger.error("Diffusion not provided")
        raise ValueError("Diffusion must be provided and not none")


def generate_latents( prompt : str, model : Any, 
diffusion : Any,
batch_size : int = LATENT_BATCH_SIZE,
guidance_scale : float = GUIDANCE_SCALE,
progress : bool = PROGRESS,
clip_denoised : bool = CLIP_DENOISED,
use_fp16 : bool = USE_FP16,
use_karras : bool = USE_KARRAS,
karras_steps : int = KARRAS_STEPS,
sigma_max : float = SIGMA_MAX,
sigma_min : float = SIGMA_MIN,
s_churn : float = S_CHURN)-> Any:
    
    validate_inputs(prompt, model, diffusion)
    logger.info(f"Inputs Verified, Starting latent generation from prompt : '{prompt}'")
    
    try:
        latents_outputs = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=progress,
        clip_denoised=clip_denoised,
        use_fp16=use_fp16,
        use_karras=use_karras,
        karras_steps=karras_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        s_churn=s_churn,
        )
        logger.info(f"LATENTS LOADED SUCCESFULLY FOR PROMPT : '{prompt}'")
        
        return latents_outputs

    except Exception as e:
        logger.exception(f"ERROR IN GENERATING LATENTS : {e}")
        raise 

    

