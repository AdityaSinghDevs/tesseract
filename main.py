from typing import Dict, Any, List

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "tesseract/core"))

from tesseract.config.config import ( USE_CUDA,FALLBACK_TO_CPU,BASE_MODEL,
                                    TRANSMITTER,DIFFUSION_CONFIG, OUTPUT_DIR,
                                    DEFAULT_FORMATS, BASE_FILE, LATENT_BATCH_SIZE,
                                    GUIDANCE_SCALE, USE_FP16, USE_KARRAS, 
                                    KARRAS_STEPS, CLIP_DENOISED,PROGRESS,
                                    SIGMA_MIN, SIGMA_MAX, S_CHURN,)
from tesseract.loggers.logger import get_logger
from tesseract.core.model_loader import get_device, load_all_models
from tesseract.core.generator import get_or_generate_latents, generate_latents
from tesseract.core.mesh_util import decode_latents, save_mesh


logger = get_logger(__name__, log_file='app.log')

def initialize_pipeline(
        use_cuda : bool=USE_CUDA,
        fallback_to_cpu : bool=  FALLBACK_TO_CPU,
         base_model :str = BASE_MODEL, transmitter: str = TRANSMITTER,
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
            "diffusion_process" : diffusion_process,
        }

        logger.info("Pipeline initiated successfully and ready for generation.")
        return pipeline_components
    except Exception as e:
        logger.error(f"Pipeline initialization failed : {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize Tesseract pipeline : {e}")
    

def generate_from_prompt(prompt:str, base_file :BASE_FILE,
                         output_dir : str = OUTPUT_DIR, formats = DEFAULT_FORMATS,
                         preloaded_pipeline: Dict[str, Any] = None, resume_latents : bool = False,
                           batch_size : int = LATENT_BATCH_SIZE,
                            guidance_scale : float = GUIDANCE_SCALE,
                            progress : bool = PROGRESS,
                            clip_denoised : bool = CLIP_DENOISED,
                            use_fp16 : bool = USE_FP16,
                            use_cuda : bool = USE_CUDA,
                            use_karras : bool = USE_KARRAS,
                            karras_steps : int = KARRAS_STEPS,
                            sigma_max : float = SIGMA_MAX,
                            sigma_min : float = SIGMA_MIN,
                            s_churn : float = S_CHURN,
                            fallback_to_cpu : bool = FALLBACK_TO_CPU) ->Dict[str, Any]:

    logger.info(f"Starting generation..")

    try:
        if not preloaded_pipeline :
            pipeline = initialize_pipeline(use_cuda = use_cuda,
        fallback_to_cpu = fallback_to_cpu)
        else :
            pipeline = preloaded_pipeline
            logger.info("Loading from preloaded pipeline..")

        text_encoder_model = pipeline["text_encoder_model"]
        transmitter_model = pipeline["transmitter"]
        diffusion_process = pipeline["diffusion_process"]

        latents = get_or_generate_latents(
            prompt=prompt,
            model=text_encoder_model,
            diffusion=diffusion_process,
            base_file=base_file,
            output_dir=output_dir,
            resume=resume_latents,
            batch_size=batch_size, guidance_scale=guidance_scale,
            progress = progress, clip_denoised=clip_denoised,
            use_fp16=use_fp16,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            s_churn=s_churn
        )

        meshes = decode_latents(model=transmitter_model, latents= latents)

        results = save_mesh(meshes=meshes, base_file=base_file,
                              output_dir=output_dir, formats=formats)
        
        logger.info(f"Generation complete for prompt : {prompt}, saved {results['count']} files.")

        return{
            "prompt" : prompt,
            "saved_files":results["saved_files"],
            "output_dir" :output_dir,
            "mesh_count" : results["count"],
            "latents_path" :  os.path.join(output_dir, "latents", f"{base_file}_latents.pt")
        }
    
    except Exception as e:
        logger.error(f"Generation failed : {e}")
        raise RuntimeError(f"Generation failed due to error : {e}")

def batch_generate(prompts: List[str], output_dir:str, base_file : str, formats = DEFAULT_FORMATS,
                   preloaded_pipeline: Dict[str, Any] = None, resume_latents : bool = False,
                           batch_size : int = LATENT_BATCH_SIZE,
                            guidance_scale : float = GUIDANCE_SCALE,
                            progress : bool = PROGRESS,
                            clip_denoised : bool = CLIP_DENOISED,
                            use_fp16 : bool = USE_FP16,
                            use_cuda : bool = USE_CUDA,
                            use_karras : bool = USE_KARRAS,
                            karras_steps : int = KARRAS_STEPS,
                            sigma_max : float = SIGMA_MAX,
                            sigma_min : float = SIGMA_MIN,
                            s_churn : float = S_CHURN,
                            fallback_to_cpu : bool = FALLBACK_TO_CPU):

        logger.info(f"Batch generation started for : {len(prompts)}")
        all_results = []

        try:
            if not preloaded_pipeline :
                pipeline = initialize_pipeline(use_cuda = use_cuda,
         fallback_to_cpu = fallback_to_cpu)
            else :
                pipeline = preloaded_pipeline
            logger.info("Loading from preloaded pipeline..")
        except Exception as e:
            logger.error(f"Failed to intialize pipeline for batch : {e}")
        
        for idx, prompt in enumerate(prompts):
            if not prompt.strip():
                logger.warning(f"Prompt {idx} is empty or invalid, Skipping..")
                continue

            file_prefix = f"{base_file}_{idx}"

            try:
                result = generate_from_prompt(
                    prompt=prompt,
                    output_dir=output_dir,
                    base_file=file_prefix,
                    formats=formats,
            base_file=base_file,
            output_dir=output_dir,
            resume=resume_latents,
            batch_size=batch_size, guidance_scale=guidance_scale,
            progress = progress, clip_denoised=clip_denoised,
            use_fp16=use_fp16,
            use_karras=use_karras,
            karras_steps=karras_steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            s_churn=s_churn)
                all_results.append(result)
                logger.info(f"[{idx+1}/{len(prompts)}] Generated for {prompt} saved successfully ")
            except Exception as e:
                logger.error(f"Failed to generate for prompt '{prompt} : {e}" , exc_info=True)
                all_results.append({
                    "prompt" : prompt,
                    "status" : "failed",
                    "error" : str(e)
                })

        logger.info(f"Batch generation completed. Success: {sum(1 for r in all_results if r.get('status','ok')!='failed')}, "f"Failed: {sum(1 for r in all_results if r.get('status')=='failed')}")

        return all_results



# if __name__ == "__main__":
#     main()