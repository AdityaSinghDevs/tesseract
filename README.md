# Tesseract V1

Generate 3D meshes from text prompts through a REST API or CLI with asynchronous job management and flexible output formats.<br>
A production-grade, modular ML pipeline that uses diffusion-driven neural nets to generate 3D mesh assets from text or image inputs, built with scalability, reliability, and deployment in mind.<br><br>
_A Mini research-to-production pipeline_

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Note on Output Quality](#note)  
4. [Installation](#installation)  
   - [Using Conda](#using-conda)  
   - [Using Python venv](#using-python-venv)  
   - [Deployment](#deployment)  
5. [Project Structure](#project-structure)  
6. [Usage](#usage)  
   - [Running the API](#running-the-api)  
     - [Local Development](#local-development)  
   - [Running via CLI](#running-via-cli)  
     - [Quick Examples](#quick-examples)  
     - [Key CLI Parameters](#key-cli-parameters)  
7. [API Examples](#api-examples)  
   - [API Documentation](#api-documentation)
8. [Configuration](#configuration)  
   - [Configuration Parameters Explained](#configuration-parameters-explained)  
     - [General Settings](#general-settings)  
     - [Device Settings](#device-settings)  
     - [Latent Generation Parameters](#latent-generation-parameters)  
     - [File Management](#file-management)  
     - [Rendering Options (Experimental)](#rendering-options-experimental)  
   - [Performance Tuning Tips](#performance-tuning-tips)  
9. [License](#license)



## Overview

Tesseract V1 exists to make the process of generating 3D meshes from text prompts accessible and scriptable. It wraps around an underlying 3D generation pipeline, exposes it through a production-ready FastAPI backend, and also provides a command-line interface for batch and local workflows.

The motivation behind Tesseract is to speed up early-stage 3D asset creation. While the generated meshes are not final production assets, they serve as useful starting points or "canvases" that can be refined further in professional 3D tools. This removes the need to begin modeling completely from scratch and allows more focus on creative iteration.

**Production-Grade Design Highlights**
- **Scalable API layer** with asynchronous job management to handle concurrent requests without blocking.
- **Modular architecture** separating core model logic, API, rendering, and configuration.
- **Config-driven execution** via YAML, enabling reproducible runs and easy tuning.
- **Structured logging** for both API and model pipelines to aid monitoring and debugging.
- **Device-aware execution** with automatic GPU/CPU fallback.
- **Multiple interface options**: REST API for integration, CLI for scripting/batch jobs.
- **Stateless API** — scalable horizontally behind a load balancer  
- Minimal external dependencies for easier deployment  


## Features

### Built for Production Tesseract V1 can:
* Generate 3D meshes directly from natural language prompts
- Export in multiple formats: **OBJ**, **PLY**, and **GLB**  
- Provide both **REST API** and **CLI** workflows  
- Support **asynchronous job creation** with background task execution  
* Offer API endpoints to check job status and download generated files
* Support extensive CLI flags for controlling batch size, formats, and other parameters
* Save all outputs in organized output directories with clear file naming
- Logging for both API and pipeline processes for monitoring and debugging 

- Handles **parallel job execution** without blocking other requests.
- Fully **stateless API design**—can be scaled horizontally behind a load balancer.
- Output **directory isolation per job** to prevent conflicts.
- Minimal external dependencies to reduce deployment friction.



## Note 
Due to the limited and not very high quality of training data for the underlying Shape-E model, outputs from Tesseract are not of final production quality. Instead, these meshes are best used as starting canvases for further refinement in modeling tools. This is an advantage over starting from a blank scene, as you immediately get a base structure to work with.  <br>

While Tesseract’s architecture is production-ready, final deployment performance depends on
the underlying model hardware, tuning, and integration with your environment.


You can check the training samples for shap-e [here](https://github.com/openai/shap-e/tree/main/samples)

It is also recommended to increase the batch size to produce more outputs in a single run, increasing the chances of finding a desirable starting point. Further tweaking of configuration parameters can also improve the usefulness of outputs and will be explained in later sections.

## Installation

It is recommended to set up Tesseract in an isolated Python environment using either Conda or venv.

### Using Conda:

```bash
conda create -n tesseract python==3.10 -y
conda activate tesseract
pip install -r requirements.txt -q 

```

### Using Python venv:

```bash
python -m venv tesseract_env
source tesseract_env/bin/activate   # On Linux or macOS
tesseract_env\Scripts\activate      # On Windows
pip install -r requirements.txt

```
- Ensure that you have Python 3.10 installed as this is the recommended version for compatibility with all dependencies.

### Deployment
Tesseract is designed for easy deployment in both development and production environments.
- **Docker-ready** (coming soon) for reproducible builds.
- Supports **GPU acceleration in cloud platforms** (AWS, GCP, Azure) and on-prem.
- Compatible with **Kubernetes scaling patterns** for serving multiple generation jobs in parallel.


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
├── shape_e_structure.svg          # Shape-E core diagram
└── LICENCE                   # Project license
```

# Usage

## Running the API

### Local Development

Start the API server using one of the following methods:

```bash
# Run with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or use the FastAPI dev runner
fastapi dev app.py
```

After startup, the following endpoints are available:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Running via CLI

### Quick Examples

```bash
# Simplest way to run 
python cli.py -p "A simple chair"

# Recommmended tweaks for good results 
python cli.py -p "A simple chair" -gs 30 --karras-steps 64 -bs 4

# Full generation with custom parameters
python cli.py -p "A simple chair" -n my_chair -o tesseract/outputs -f ply glb -bs 2 -gs 20 --karras-steps 64 --use-fp16 --use-karras

# Batch processing from file
python cli.py -b prompts.txt -o output_folder -f obj glb --karras-steps 25

# Single prompt with dry run (testing configuration)
python cli.py -p "A simple chair" --dry-run
```

### Key CLI Parameters

| Flag | Description |
|------|-------------|
| `-p, --prompt` | Single text prompt to generate a 3D mesh |
| `-b, --batch_file` | Path to text file with one prompt per line |
| `-f, --formats` | Output formats: `ply`, `obj`, `glb` (default: ply) |
| `-n, --base_file` | Base filename for output files (default: generated_mesh) |
| `-bs, --batch-size` | Number of shapes(outputs) per prompt (default: 1) |
| `-gs, --guidance-scale` | Prompt adherence strength (default: 12.0) |
| `--karras-steps` | Denoising steps for quality (default: 30) |
| `--use-fp16` | Enable half-precision for memory efficiency (Default : On) |
| `-r, --resume-latents` | Resume from cached latents if available |
| `--dry-run` | Test configuration without generating files |

<!-- Use `--help` to see all available commands and their default values. -->

- Further details about each configuration is given in [this](!Configuration) section

*Use `--help` or `-h` with CLI to see the complete list of available options and their current default values.*

#### Example 
`python cli.py --help`

## API Examples

### Example API Calls

```bash
# Submit a generation job
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A stylized wooden bench",
    "base_file": "bench_v1",
    "formats": ["ply"],
    "resume_latents": false
  }'

# Check job status
curl "http://127.0.0.1:8000/api/v1/status/<job_id>"

# Download generated meshes as ZIP
curl -O -J "http://127.0.0.1:8000/api/v1/download/<job_id>"
```

### API Documentation

- **Interactive Docs**: [Swagger UI](http://127.0.0.1:8000/docs)
- **Static Docs**: [ReDoc](http://127.0.0.1:8000/redoc)

## Configuration

Tesseract uses `defaults.yaml` for runtime configuration. Modify these settings to tune generation behavior and resource usage.
You can refer to `defaults.yaml` file for understanding detailed effects of each parameter.

### Configuration Parameters Explained

#### General Settings
- **`project_name`**: Project identifier for logs and metadata
- **`model`**: Underlying model family (`shap-e`)
- **`base_model`**: Text-conditioned model variant (`text300M`)
- **`transmitter`**: Renderer model identifier
<!-- - **`seed`**: Random seed for reproducible results -->

#### Device Settings
- **`use_cuda`**: Enable CUDA acceleration when available
- **`fallback_to_cpu`**: Allow CPU fallback if CUDA unavailable

#### Latent Generation Parameters
- **`batch_size`**: Range `[1-8+]` - Higher values increase memory usage
- **`guidance_scale`**: Range `[1.0-20.0+]` - Controls prompt fidelity vs creativity
- **`karras_steps`**: Range `[15-128+]` - More steps = higher quality, slower generation
- **`sigma_min/max`**: Noise level bounds affecting detail vs noise tradeoff
- **`s_churn`**: Range `[0.0-10.0]` - Adds randomness/diversity to sampling

#### File Management
- **`output_dir`**: Directory for generated meshes and assets
- **`base_file`**: Default filename template
- **`default_format`**: Supported formats: `ply`, `obj`, `glb`

#### Rendering Options (Experimental)
- **`render_mode`**: Preview rendering engine (`nerf`)
- **`size`**: Preview resolution for images/GIFs
- **`render`**: Enable/disable automatic preview generation

### Performance Tuning Tips

**For Limited GPU Memory:**
- Set `batch_size: 1`
- Start with `karras_steps: 20-25`
- Enable `use_fp16: true`

**For Quality vs Speed:**
- **Higher Quality**: Increase `karras_steps` (50-100), `guidance_scale` (20+)
- **Faster Generation [NOT RECOMMENDED]**: Decrease `karras_steps` (10-15), `guidance_scale` (8-12)

**For Creative vs Faithful Output:**
- **More Creative**: Lower `guidance_scale` (5-10)
- **More Faithful**: Higher `guidance_scale` (15-25)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

You are free to:

• Use, modify, and distribute this software for personal, academic, or commercial purposes  
• Clone it for research, testing, or improvement  
• Run it locally or in production environments  

You must:

• Keep the license intact in all copies or substantial portions of the software  
• Release source code for any modifications you make if you distribute or run it as a network service  
• Comply with the licensing terms of any third-party dependencies used in this project  

You cannot:

• Make proprietary or closed-source derivatives without also releasing the modified source code  
• Remove copyright or license notices  

This project makes use of [Shap-E](https://github.com/openai/shap-e), an OpenAI project, which is released under the MIT License.  
All usage of Shap-E within this project must follow the guidelines and licensing terms provided by OpenAI.  

For the full license text, see the `LICENSE` file in this repository.
