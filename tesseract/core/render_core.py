
from typing import Any
import sys
import webbrowser
import tempfile
import torch

from .shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

from ..loggers.logger import get_logger
from ..config.config import RENDER_MODE,RENDER_SIZE,TRANSMITTER

# logger = get_logger(__name__, log_file="app.log")

def validate_render_inputs(device, latents, size):
    if not isinstance(device, torch.device):
        raise TypeError(f"Expected torch.device, got {type(device)}")

    if latents is None or len(latents) == 0:
        raise ValueError("No latents provided for rendering.")

    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"Render size must be a positive int, got {size}")
    
def in_notebook() -> bool:
    """Detect if running in Jupyter Notebook or Lab."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'  # Jupyter/Colab
    except Exception:
        return False
    
def in_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # type: ignore
        return True
    except ImportError:
        return False

def render_image(device : torch.device,
                 latents : Any,
                 transmitter,
                 size : int = RENDER_SIZE,
                 render_mode : str = RENDER_MODE,
                 )-> list[str]:
    
    validate_render_inputs(device, latents, size)
    # logger.info("Inputs for Rendering Validated")
    
    # logger.info("Initializing Rendering..")

    try:
        cameras = create_pan_cameras(size=size,
                                     device=device)
        # logger.info("Cameras set successfully")
    except Exception as e:
        # logger.error(f"Unable to setup cameras : {e}")
        raise RuntimeError(f"Camera initialization failed : {str(e)}" )
    
    html_outputs = []
    
    for i, latent in enumerate(latents):
        try:
            images = decode_latent_images(xm=transmitter,
                                        latent=latent,
                                        rendering_mode=render_mode, cameras=cameras)
            # output_files.append(images)
        except Exception as e:
            print(f"[ERROR] Decoding latent {i} failed: {e}")
            continue

        widget  = gif_widget(images)


        if in_notebook() or in_colab(): 
            from IPython.display import display
            # html_widget = HTML(widget.value)
            # display(html_widget)
            # output_files.append(html_widget)
            # logger.info(f"Latent {i} rendered in notebook")
            display(widget)
            continue

        elif sys.stdout.isatty() and not in_colab():
            try:
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
                    f.write(widget.value)
                    temp_html = f.name
                    webbrowser.open(f"file://{temp_html}")
                # logger.info(f"Opened latent {i} in web browser.")
            except Exception as e:
                print(f"[ERROR] Failed to open browser preview: {e}")

    return html_outputs
