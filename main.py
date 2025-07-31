import argparse
from typing import Dict, Any, List

from tesseract.config.config import ( USE_CUDA,FALLBACK_TO_CPU,BASE_MODEL,TRANSMITTER,DIFFUSION_CONFIG, OUTPUT_DIR, DEFAULT_FORMATS, BASE_FILE)
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
            "diffusion_process" : diffusion_process,
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
        diffusion_process = pipeline["diffusion_process"]

        latents = generate_latents(prompt=prompt, model=text_encoder_model,
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

def batch_generate(prompts: List[str], output_dir:str, base_file : str):

        logger.info(f"Batch generation started for : {len(prompts)}")
        all_results = []
        
        for idx, prompt in enumerate(prompts):
            if not prompt.strip():
                logger.warning(f"Prompt {idx} is empty or invalid, Skipping..")
                continue

            file_prefix = f"{base_file}_{idx}"

            try:
                result = generate_from_prompt(prompt=prompt, output_dir=output_dir, base_file=base_file)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="TesseractV1 - Text-to-3D generation using Shap-E"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "-p","--prompt",
        type=str,
        help="Single prompt to generate a 3D mesh"
    )

    group.add_argument(
        "-b","--batch_file",
        type=str,
        help="Path to a text file with one prompt per line"
        )
    
    parser.add_argument(
        "-o","output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save generated files (default : {OUTPUT_DIR})"
    )

    parser.add_argument(
        "-f","--formats",
        type=str,
        nargs="+",
        default=DEFAULT_FORMATS,
        choices=["ply", "obj", "glb"],
        help="Mesh formats to export (default : ply)"
    )

    parser.add_argument(
        "-n","--base-file",
        type=str,
        default={BASE_FILE},
        help = f"Base name for output files (default : {BASE_FILE})"
    )
    return parser.parse_args()


