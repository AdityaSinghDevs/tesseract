import sys
import os
import argparse

from tesseract.loggers.logger import get_logger
from tesseract.config.config import OUTPUT_DIR, DEFAULT_FORMATS, BASE_FILE
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
        help="Run pipeline without generating or saving any files (for testing)"
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
                                                formats=args.formats )
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
                formats=args.formats
                )
            
            
            print(f"\n Batch generation complete : {len(results)} prompts processed")

            for res in results:
                print(f"-Prompt: {res['prompt']}-> {len(res['saved_files'])} files saved")

    except Exception as e:
        logger.error(f"CLI execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__": main()