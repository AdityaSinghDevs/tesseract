
from typing import Any
import os
import sys
import webbrowser
import tempfile
import torch

from .shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

from ..loggers.logger import get_logger
from ..config.config import RENDER_MODE,RENDER_SIZE,TRANSMITTER

logger = get_logger(__name__, log_file="app.log")

def validate_render_inputs(device, latents, size):
    if not isinstance(device, torch.device):
        raise TypeError(f"Expected torch.device, got {type(device)}")

    if latents is None or len(latents) == 0:
        raise ValueError("No latents provided for rendering.")

    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Render size must be a positive int, got {size}")

def render_image(device : torch.device,
                 latents : Any,
                 transmitter : str = TRANSMITTER,
                 size : int = RENDER_SIZE,
                 render_mode : str = RENDER_MODE,
                 cli_preview : bool = True,
                 )->Any :
    
    validate_render_inputs(device, latents, size)
    logger.info("Inputs for Rendering Validated")
    
    logger.info("Initializing Rendering..")

    try:
        cameras = create_pan_cameras(size=size,
                                     device=device)
        logger.info("Cameras set successfully")
    except Exception as e:
        logger.error(f"Unable to setup cameras : {e}")
        raise RuntimeError(f"Camera initialization failed : {str(e)}" )
    
    # output_files = []
    
    for i, latent in enumerate(latents):
        try:
            images = decode_latent_images(xm=transmitter,
                                        latent=latent,
                                        rendering_mode=render_mode)
            # output_files.append(images)
        except Exception as e:
            logger.error(f"Decoding latent {i} failed: {e}")
            continue

        widget  = gif_widget(images)


        try : 
            get_ipython #type: ignore
            from IPython.display import display
            display(widget)
            logger.info(f"Latent {i} rendered in notebook")
        except NameError:
            pass

        if cli_preview and sys.stdout.isatty():
            try:
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
                    f.write(widget._repr_html_())
                    temp_html = f.name
                    webbrowser.open(f"file://{temp_html}")
                logger.info(f"Opened latent {i} in web browser.")
            except Exception as e:
                logger.error(f"Failed to opem browser preview : {e}")
