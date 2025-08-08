# Tesseract V1

A 3D generation toolkit that transforms text prompts into editable mesh formats

## Overview and Introduction

Tesseract V1 exists to make the process of creating 3D meshes from scratch faster and more accessible. While most artists and developers spend significant time starting from a blank canvas, Tesseract lets you describe what you need in words and produces a starting 3D mesh in multiple formats. This approach is ideal when you want a rough base that you can refine instead of modeling from zero

The system is built on top of the shap-e model and integrates it with a production-grade FastAPI backend and a feature-rich CLI. It supports async background jobs, multiple mesh formats, and status polling for efficient workflows

Key features include
- Text prompt to mesh generation with configurable parameters
- Export to PLY OBJ and GLB formats
- REST API with status polling
- CLI support with extensive flags and configuration options
- Async job handling and background task management
- Configurable output structure for API and CLI generated files

Note  
Due to the limited and not very high quality training data of the underlying shap-e model the outputs of Tesseract are not perfect. These meshes are best used as base canvases for further refinement in your preferred 3D software. This still offers a significant edge over starting from scratch. For better chances of achieving a desirable starting point consider increasing the batch size to generate more outputs and adjusting configuration parameters as explained in later sections

## Features

Tesseract V1 offers
- 3D mesh generation from text prompts
- Export to PLY OBJ and GLB formats
- REST API with job submission and status polling
- CLI interface with multiple flags and options
- Async processing for long running generation jobs
- Background task handling for scalable workflows

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