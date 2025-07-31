import os
from typing import Dict, Any, List

from tesseract.config.config import ( USE_CUDA,FALLBACK_TO_CPU,BASE_MODEL,TRANSMITTER,DIFFUSION_CONFIG, OUTPUT_DIR, DEFAULT_FORMATS)
from tesseract.loggers.logger import get_logger
from tesseract.core.model_loader import get_device, load_all_models
from tesseract.core.generator import generate_latents
from tesseract.core.mesh_util import decode_latents, save_mesh

logger = get_logger(__name__, log_file='app.log')

def initialize_pipeline(
        use_cuda : bool=USE_CUDA,
        fallback_to_cpu : bool=  FALLBACK_TO_CPU,
        device = str, base_model :str = BASE_MODEL, transmitter: str = TRANSMITTER,
diffusion_config : str = DIFFUSION_CONFIG
) -> Dict[str, Any]:

    logger.info("Initializing Tesseract pipeline")

    try:
        device = get_device(use_cuda, fallback_to_cpu)
        logger.info(f"Using device {device}")

        transmitter_model, text_encoder_model, diffusion_process = load_all_models(device=device, base_model=base_model, transmitter=transmitter, diffusion_config=diffusion_config)
        logger.info("All models loaded successfully.")

        pipeline_components={
            "transmitter" : transmitter_model,
            "text_encoder_model" : text_encoder_model,
            "diffusion" : diffusion_process,
        }

        logger.info("Pipeline initiated successfully and ready for generation.")
        return pipeline_components
    except Exception as e:
        logger.error(f"Pipeline initialization failed : {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Tesseract pipeline : {e}")
    

def generate_from_prompt(prompt:str, base_file :str,
                         output_dir : str = OUTPUT_DIR, format = DEFAULT_FORMATS,
                         preloaded_pipeline: Dict[str, Any] = None,) ->Dict[str, Any]:

    logger.info(f"Starting generation..")

    try:
        if not preloaded_pipeline :
            pipeline = initialize_pipeline()
        else :
            pipeline = preloaded_pipeline
            logger.info("Loading from preloaded pipeline..")

        text_encoder_model = pipeline["text_encoder_model"]
        transmitter_model = pipeline["transmitter"]
        diffusion_process = pipeline["diffusion"]

        latents = generate_latents(prompt=prompt,         model=text_encoder_model,
                                   diffusion=diffusion_process )
        
        meshes = decode_latents(model=transmitter_model, latents= latents)

        results = save_mesh(meshes=meshes, base_file=base_file,
                              output_dir=output_dir, formats=format)
        
        logger.info(f"Generation complete for prompt : {prompt}, saved {results['count']} files.")

        return{
            "prompt" : prompt,
            "saved_files":results["saved_files"],
            "output_dir" :output_dir,
            "mesh_count" : results["count"]
        }
    
    except Exception as e:
        logger.error(f"Generation failed : {e}")
        raise RuntimeError(f"Generation failed due to error : {e}")

