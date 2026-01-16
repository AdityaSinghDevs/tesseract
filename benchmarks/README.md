### Note on Result File Naming

Early benchmark runs used a simpler file naming scheme.
After benchmarking logic stabilized, result filenames were standardized to:

{benchmark}_{device}_{config}_batch_size_{N}_{prompt}.json

Along with better logging of parameters.
For consistency, earlier result files were renamed without modifying their contents.
All metrics inside JSON files are produced directly by the benchmark scripts.


# Benchmarks

This directory contains the benchmarking infrastructure and experimental results for **Tesseract**, focusing on inference-time performance, scalability, and resource utilization. The goal of these benchmarks is to provide **transparent, reproducible, and configuration-controlled** measurements of model inference under different runtime settings.

---

## Overview

The benchmarking framework evaluates the following aspects:

- **Inference latency vs end-to-end latency**
- **GPU vs CPU performance**
- **GPU memory utilization**
- **Batch size scaling**
- **Sampling step scaling**
- **Prompt complexity effects**

All benchmarks are executed using a fixed pipeline and configuration system to ensure fair comparisons across runs.

---

## Benchmarking Script

All measurements are performed using a single unified script:
```
benchmarks/scripts/benchmark_inference.py
```

This script is intentionally designed to be **parameter-driven**, with runtime behavior controlled via:

- device selection (`cpu` / `gpu`)
- batch size
- sampling configuration (`baseline`, `high_cost`)
- prompt selection (`simple`, `medium`, `complex`)

This avoids duplicated logic and ensures consistency across experiments.

---

## Configurations

Benchmark configurations are defined in:
```
benchmarks/configs/
```

### Available configs

- **baseline**
  - Standard inference configuration
  - 15 sampling steps
  - Used for most comparative experiments

- **high_cost**
  - Increased sampling budget
  - 64 sampling steps
  - Used to study sampling-time scaling behavior

All configs are loaded dynamically using the configuration loader and recorded in the benchmark outputs.

---

## Prompts

To study prompt complexity effects, three fixed prompts are used:
```
benchmarks/prompts/
├── simple.txt   # e.g. "A chair"
├── medium.txt   # e.g. "A shark"
└── complex.txt  # e.g. "A detailed spaceship"
```

These prompts are intentionally short and deterministic to isolate **pipeline overhead** rather than semantic variability.

---

## Metrics Collected

For each benchmark run, the following metrics are recorded:

- **Inference latency (seconds)**  
  Time spent exclusively in latent generation.

- **End-to-end latency (seconds)**  
  Includes inference, decoding, and mesh serialization.

- **Peak GPU memory (bytes)**  
  Recorded using `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()` when running on GPU. For CPU runs, this field is recorded as `null`.

Each experiment is repeated **three times**, and results are reported as:
```
mean ± sample standard deviation
```

---

## Results Storage

Raw benchmark outputs are stored as JSON files in:
```
benchmarks/results/raw/
```

File names follow a consistent schema:
```
{benchmark_name}_{device}_{config}_batch_size_{BATCH_SIZE}_{PROMPT}.json
```

Example:
```
inference_latency_gpu_baseline_batch_size_1_simple.json
```

This naming convention ensures traceability and reproducibility.

---

## Tables and Aggregated Results

Aggregated results and statistical summaries are stored in:
```
benchmarks/results/tables/
```

These tables are derived directly from the raw JSON outputs and include:

- GPU vs CPU comparisons
- Batch size scaling
- Sampling step scaling
- Prompt complexity analysis

Final benchmark summaries and interpretations are documented in:
```
PERFORMANCE.md
```

---

## Reproducibility

To reproduce any benchmark:

1. Select the desired configuration:
   - device (`cpu` or `gpu`)
   - batch size
   - config variant (`baseline` / `high_cost`)
   - prompt (`simple`, `medium`, `complex`)

2. Run:
```bash
python -m benchmarks.scripts.benchmark_inference 

```

Ensure the environment details (Python, PyTorch, CUDA, hardware) are recorded.

A sample Colab notebook demonstrating this is provided in:
```
benchmarks/notebooks/
```

---

## Notes

- GPU benchmarks were executed on NVIDIA Tesla T4
- CPU benchmarks are significantly slower and primarily included for reference
- FP16 inference is enabled consistently across GPU runs
- No results are manually edited; all reported statistics are derived from logged outputs

---

## Summary

This benchmarking setup prioritizes:

- **correctness** over over-engineering
- **clarity** over excessive abstraction
- **reproducibility** over convenience

It reflects real-world ML systems benchmarking practices and is intended to serve both research and engineering evaluation needs.