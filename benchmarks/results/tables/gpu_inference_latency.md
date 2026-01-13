# GPU Inference Latency Benchmark

This document summarizes GPU-based inference latency measurements for Tesseract v1 under a controlled configuration.  
The goal is to characterize steady-state latency, isolate inference cost from system overhead, and observe GPU memory behavior.

---

## Benchmark Configuration

- **Device:** GPU (NVIDIA Tesla T4)
- **Batch Size:** 1
- **Sampling Steps:** 15
- **Prompt:** “A chair”
- **Runs:** 3 (after warm-up)
- **Measurement Mode:**
  - Inference-only latency measures latent generation only
  - End-to-end latency includes inference, decoding, and mesh serialization

---

## Per-Run Results

| Run | End-to-End Latency (s) | Inference-only Latency (s) | Peak GPU Memory (GB) |
|----:|-----------------------:|---------------------------:|---------------------:|
| 0 | 6.14 | 4.92 | 5.05 |
| 1 | 6.01 | 4.93 | 5.06 |
| 2 | 6.05 | 4.97 | 5.06 |

---

## Aggregated Statistics

| Metric | Mean |
|------|------:|
| End-to-End Latency | **6.07 s** |
| Inference-only Latency | **4.94 s** |
| Inference Share of End-to-End | **~82%** |
| Peak GPU Memory | **5.06 GB** |

---

## Observations

- **Inference dominates latency:** Diffusion-based latent generation accounts for approximately 82% of total end-to-end latency.
- **Stable execution:** Run-to-run variance is minimal, indicating effective warm-up and steady-state measurement.
- **Bounded system overhead:** Non-inference overhead (decoding, serialization, I/O) remains consistent at ~1.1 s.
- **Predictable memory usage:** Peak GPU memory remains stable across runs, with no evidence of memory leakage or fragmentation.

---

## Notes

- Measurements reflect steady-state behavior after an explicit warm-up run.
- No optimization, caching, or concurrency was applied.
- These results serve as a baseline for comparison against CPU execution and batch-size scaling benchmarks.
