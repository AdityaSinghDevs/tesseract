import sys
import os
import argparse

from tesseract.loggers.logger import get_logger
from tesseract.config.config import ( USE_CUDA,FALLBACK_TO_CPU, OUTPUT_DIR,
                                    DEFAULT_FORMATS, BASE_FILE, LATENT_BATCH_SIZE,
                                    GUIDANCE_SCALE, USE_FP16, USE_KARRAS, 
                                    KARRAS_STEPS, CLIP_DENOISED,PROGRESS,
                                    SIGMA_MIN, SIGMA_MAX, S_CHURN,RENDER_INSTANCE)
from main import generate_from_prompt, batch_generate



logger = get_logger(__name__, log_file='app.log')


def parse_args():
    parser = argparse.ArgumentParser(
        description="TesseractV1 - Text-to-3D generation using Shap-E"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
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
        "-o","--output_dir",
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
        "-n","--base_file",
        type=str,
        default=BASE_FILE,
        help = f"Base name for output files (default : {BASE_FILE})"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without generating or saving any files "
        "(for testing)"
    )

    parser.add_argument(
        "-r", "--resume-latents",
        action="store_true",
        help="If set, tries to resume from existing cached latents before generating new one"
    )

    parser.add_argument(
        "-bs", "--batch-size",
        type=int,
        default=LATENT_BATCH_SIZE,
        help=f"Batch size for latent generation (default: {LATENT_BATCH_SIZE})"
    )
    parser.add_argument(
        "-gs", "--guidance-scale",
        type=float,
        default=GUIDANCE_SCALE,
        help=f"Guidance scale for diffusion (default: {GUIDANCE_SCALE})"
    )
    parser.add_argument(
        "--progress",
        action="store_true" if not PROGRESS else "store_false",
        default=PROGRESS,
        help=f"Show progress bar (default: {PROGRESS})"
    )
    parser.add_argument(
        "--clip-denoised",
        action="store_true" if not CLIP_DENOISED else "store_false",
        default=CLIP_DENOISED,
        help=f"Clip denoised samples during generation (default: {CLIP_DENOISED})"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true" if not USE_FP16 else "store_false",
        default=USE_FP16,
        help=f"Use FP16 for faster inference (default: {USE_FP16})"
    )
    parser.add_argument(
        "--use-karras",
        action="store_true" if not USE_KARRAS else "store_false",
        default=USE_KARRAS,
        help=f"Use Karras noise schedule (default: {USE_KARRAS})"
    )
    parser.add_argument(
        "--karras-steps",
        type=int,
        default=KARRAS_STEPS,
        help=f"Number of Karras steps (default: {KARRAS_STEPS})"
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=SIGMA_MAX,
        help=f"Maximum sigma for Karras schedule (default: {SIGMA_MAX})"
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=SIGMA_MIN,
        help=f"Minimum sigma for Karras schedule (default: {SIGMA_MIN})"
    )
    parser.add_argument(
        "--s-churn",
        type=float,
        default=S_CHURN,
        help=f"Sigma churn parameter (default: {S_CHURN})"
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=USE_CUDA,
        help=f"Force CUDA usage if available (default: {USE_CUDA})")
    
    parser.add_argument(
        "--fallback-to-cpu",
        action="store_true", 
        default=FALLBACK_TO_CPU,
        help=f"Fallback to CPU if CUDA is unavailable (default: {FALLBACK_TO_CPU})")
    
    parser.add_argument(
        "--render",
        action="store_true",
        default=RENDER_INSTANCE,
        help=f"Show rendered outputs in notebook or web browser (default : {RENDER_INSTANCE})"
    )


    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("CLI execution started")

    try:
        if args.prompt:

            if args.dry_run:
                logger.info(f"DRY-RUN: would generate mesh for prompt '{args.prompt}'")
                print(f"[DRY-RUN] Prompt: {args.prompt} → No files generated")
                return
            
            
            result = generate_from_prompt(prompt=args.prompt,
                                        base_file=args.base_file,
                                        output_dir=args.output_dir,
                                        formats=args.formats, 
                                        resume_latents=args.resume_latents,
                                        batch_size=args.batch_size,
                                        guidance_scale=args.guidance_scale,
                                        progress=args.progress,
                                        clip_denoised=args.clip_denoised,
                                        use_fp16=args.use_fp16,
                                        use_cuda=args.use_cuda,
                                        use_karras=args.use_karras,
                                        karras_steps=args.karras_steps,
                                        sigma_max=args.sigma_max,
                                        sigma_min=args.sigma_min,
                                        s_churn=args.s_churn,
                                        fallback_to_cpu=args.fallback_to_cpu,
                                        render=args.render)
            print(f"\n Generated mesh for prompt : '{args.prompt}'")
            print(f"\n Saved files : {result['saved_files']}\n")
            
        elif args.batch_file:
            if not os.path.exists(args.batch_file):
                logger.error("Batch file does not exist")
                sys.exit(1)

            with open(args.batch_file, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]

            if args.dry_run:
                 print(f"[DRY-RUN] Would process {len(prompts)} prompts:")
                 for idx, p in enumerate(prompts, 1):
                     print(f"{idx}.{p}")
                     sys.exit(0)

            results = batch_generate(
                prompts=prompts, 
                output_dir=args.output_dir, 
                base_file=args.base_file,
                formats=args.formats,
                resume_latents=args.resume_latents,
                                        batch_size=args.batch_size,
                                        guidance_scale=args.guidance_scale,
                                        progress=args.progress,
                                        clip_denoised=args.clip_denoised,
                                        use_fp16=args.use_fp16,
                                        use_cuda=args.use_cuda,
                                        use_karras=args.use_karras,
                                        karras_steps=args.karras_steps,
                                        sigma_max=args.sigma_max,
                                        sigma_min=args.sigma_min,
                                        s_churn=args.s_churn,
                                        
                                        fallback_to_cpu=args.fallback_to_cpu,
                )
            
            
            print(f"\n Batch generation complete : {len(results)} prompts processed")

            for res in results:
                print(f"-Prompt: {res['prompt']}-> {len(res['saved_files'])} files saved")

    except Exception as e:
        logger.error(f"CLI execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__": main()