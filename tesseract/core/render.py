
from .shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget

from ..loggers.logger import get_logger
from ..config.config import RENDER_MODE,RENDER_SIZE,TRANSMITTER

logger = get_logger(__name__, log_file="app.log")

def render_image(latents : )