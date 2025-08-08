# Tesseract V1

Generate 3D meshes from text prompts through a REST API or CLI with asynchronous job management and flexible output formats.

## Overview

Tesseract V1 exists to make the process of generating 3D meshes from text prompts accessible and scriptable. It wraps around an underlying 3D generation pipeline, exposes it through a production-ready FastAPI backend, and also provides a command-line interface for batch and local workflows.

The motivation behind Tesseract is to speed up early-stage 3D asset creation. While the generated meshes are not final production assets, they serve as useful starting points or "canvases" that can be refined further in professional 3D tools. This removes the need to begin modeling completely from scratch and allows more focus on creative iteration.

Key features include:

* Support for multiple mesh formats such as OBJ, PLY, and GLB
* REST API with asynchronous job management and status polling
* Command-line interface with extensive flag support
* Configurable output directories and generation parameters
* Logging for both API and pipeline processes
* Ready to run in both development and production environments

Under the hood, Tesseract uses a modified pipeline built on the <b>Shape-E</b> model and FastAPI for its backend. Python’s async capabilities are leveraged for background task execution so that generation jobs do not block incoming requests.

## Features

Tesseract V1 can:
* Generate 3D meshes directly from natural language prompts
* Export meshes in OBJ, PLY, and GLB formats
* Provide both REST API and CLI-based workflows
* Allow asynchronous job creation with background task execution
* Offer API endpoints to check job status and download generated files
* Support extensive CLI flags for controlling batch size, formats, and other parameters
* Save all outputs in organized output directories with clear file naming
* Log both job-level and system-level activity for debugging and monitoring

## Note 
Due to the limited and not very high quality of training data for the underlying Shape-E model, outputs from Tesseract are not of final production quality. Instead, these meshes are best used as starting canvases for further refinement in modeling tools. This is an advantage over starting from a blank scene, as you immediately get a base structure to work with.  

You can check the training samples for shap-e [here](!https://github.com/openai/shap-e/tree/main/samples)

It is also recommended to increase the batch size to produce more outputs in a single run, increasing the chances of finding a desirable starting point. Further tweaking of configuration parameters can also improve the usefulness of outputs and will be explained in later sections.

## Installation

It is recommended to set up Tesseract in an isolated Python environment using either Conda or venv.

### Using Conda:

```
conda create -n tesseract python==3.10 -y
conda activate tesseract
pip install -r requirements.txt -q 

```

### Using Python venv:

```
python -m venv tesseract_env
source tesseract_env/bin/activate   # On Linux or macOS
tesseract_env\Scripts\activate      # On Windows
pip install -r requirements.txt

```
- Ensure that you have Python 3.10 installed as this is the recommended version for compatibility with all dependencies.


## Project Structure

```
tesseract/
├── api/
│   ├── __init__.py
│   ├── api.py                # API implementation
│   └── schemas.py            # Pydantic API request/response schemas
├── tesseract/
│   ├── config/               # Configuration files and settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── generator.py      # Core generation logic
│   │   ├── mesh_util.py      # Mesh processing utilities
│   │   ├── model_loader.py   # Model loading and management
│   │   └── render_core.py    # Core rendering functionality
│   ├── loggers/              # Logging configuration and utilities
│   ├── api_outputs/          # API-generated output files
│   └── outputs/              # CLI generated output files
├── logs/                     # Application logs (gitignored)
├── notebooks/                # Jupyter Notebook samples for using project in Colab or similar
├── app.py                    # API entry point (FastAPI)
├── cli.py                    # CLI entry point
├── main.py                   # Main application logic
├── render.py                 # Rendering script (under development)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── shape_e_core.svg          # Shape-E core diagram
└── LICENCE                   # Project license
```












# tesseract
A modular ML pipeline that uses diffusion driven neural nets, to generate usable 3D Mesh assets from text or image inputs.
 mini research-to-production pipeline

 (tesseract) PS C:\Users\Aditya Pratap Singh\OneDrive\Desktop\Codes\tesseract_project> python cli.py --help
usage: cli.py [-h] (-p PROMPT | -b BATCH_FILE) [-o OUTPUT_DIR]
              [-f {ply,obj,glb} [{ply,obj,glb} ...]] [-n BASE_FILE] [--dry-run]
              [-r] [-bs BATCH_SIZE] [-gs GUIDANCE_SCALE] [--progress]
              [--clip-denoised] [--use-fp16] [--use-karras]
              [--karras-steps KARRAS_STEPS] [--sigma-max SIGMA_MAX]
              [--sigma-min SIGMA_MIN] [--s-churn S_CHURN] [--use-cuda]
              [--fallback-to-cpu]

TesseractV1 - Text-to-3D generation using Shap-E

options:
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Single prompt to generate a 3D mesh
  -b BATCH_FILE, --batch_file BATCH_FILE
                        Path to a text file with one prompt per line
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory to save generated files (default :
                        tesseract/outputs)
  -f {ply,obj,glb} [{ply,obj,glb} ...], --formats {ply,obj,glb} [{ply,obj,glb} ...]
                        Mesh formats to export (default : ply)
  -n BASE_FILE, --base_file BASE_FILE
                        Base name for output files (default : generated_mesh)
  --dry-run             Run pipeline without generating or saving any files (for
                        testing)
  -r, --resume-latents  If set, tries to resume from existing cached latents before
                        generating new one
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for latent generation (default: 1)
  -gs GUIDANCE_SCALE, --guidance-scale GUIDANCE_SCALE
                        Guidance scale for diffusion (default: 12.0)
  --progress            Show progress bar (default: True)
  --clip-denoised       Clip denoised samples during generation (default: True)
  --use-fp16            Use FP16 for faster inference (default: True)
  --use-karras          Use Karras noise schedule (default: True)
  --karras-steps KARRAS_STEPS
                        Number of Karras steps (default: 30)
  --sigma-max SIGMA_MAX
                        Maximum sigma for Karras schedule (default: 160.0)
  --sigma-min SIGMA_MIN
                        Minimum sigma for Karras schedule (default: 0.001)
  --s-churn S_CHURN     Sigma churn parameter (default: 0.0)
  --use-cuda            Force CUDA usage if available (default: True)
  --fallback-to-cpu     Fallback to CPU if CUDA is unavailable (default: True)