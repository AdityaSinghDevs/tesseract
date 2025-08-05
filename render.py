import argparse 
import os 

import torch

from tesseract.core.render_core import render_image
from main import initialize_pipeline
from tesseract.config.config import RENDER_MODE, RENDER_SIZE, USE_CUDA,FALLBACK_TO_CPU

def main():
    parser = argparse.ArgumentParser(description="Render latent 3D models as GIFs")

    parser.add_argument(
        "--latents",
        type=str,
        required=True,
        help = "Path to .pt latents file")
    
    parser.add_argument(
        "--render-mode" ,
        type = str,
        default=RENDER_MODE,
        help=f"Rendering mode: nerf/stf (default : {RENDER_MODE})"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=RENDER_SIZE,
        help=f"Size of GIFs (defualt : {RENDER_SIZE})")
    
    args = parser.parse_args()

    models = initialize_pipeline(use_cuda = USE_CUDA,
        fallback_to_cpu = FALLBACK_TO_CPU)

    transmitter_model = models["transmitter"]

    latents = torch.load(args.latents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Rendering {len(latents)} latents from {args.latents}...")
    html_files = render_image(device, latents, transmitter  = transmitter_model, size=args.size, render_mode=args.render_mode)
    print(f"[INFO] Rendering complete. HTML previews: {html_files}")