## Performance Goals

The performance goals of Tesseract are focused on understanding and characterizing system behavior rather than minimizing raw latency.

The primary goals are:

1. End-to-end latency transparency across the request lifecycle.
2. Performance quantification between GPU-accelerated and CPU-fallback execution.
3. Effect of configuration parameters and batch size on latency and throughput.
4. Memory usage patterns during inference, particularly peak GPU memory consumption.
5. Making the tradeoffs between portability and performance explicit.

These goals emphasize measurement, reproducibility, and interpretability over aggressive system-level optimization.

## Key Performance Questions
1. What portion of end-to-end latency is attributable to model inference vs non-inference overhead?
2. How much slower is CPU fallback compared to GPU execution for the same configuration?
3. How does batch size and selected configuration parameters(eg., sampling steps) affect latency and throughput?
4. Where does GPU memory usage peak during execution?

## Metrics

| Performance Question | Metric | Rationale |
|:--------------------|:-------|:----------|
| Inference vs Overhead| End-to-end latency, inference-only latency | Separates model execution time from system-level overhead, indicating whether optimization efforts should target the model runtime or surrounding infrastructure. |
| CPU vs GPU gap | End-to-end latency per device | Quantifies cost of CPU fallback under identical configurations. |
| Batch and selected configuration scaling | Latency and throughput | Characterizes the trade-off between per-request latency and aggregate throughput as batch size and selected configuration parameters vary. |
| Memory pressure | Peak GPU memory | Identifies memory bottlenecks and OOM risk, GPU memory metrics are collected only for GPU runs, for CPU runs, this field is recorded as null.|

All latency metrics are reported as mean ± sample standard deviation over three independent runs.

## Test Matrix

Benchmarks are executed across a controlled set of configurations to ensure fair comparison and avoid cherry-picking results.

| Fixed parameters | Variable parameters  |
|:-----------------|:---------------------|
| Prompt set: simple, medium, complex | *Device:* GPU, CPU |
| Output format: PLY | *Batch size:* 1, 2, 4 |
| Model variant: fixed | *Sampling steps:* 15 (default, baseline), 64 (high-cost reference) |
| Precision: default model precision | |
| Environment : fixed per benchmark run | |   

## Explicit Non-Goals

| Non-Goal | Reason |
|:---------|:-------|
| Multi-node or distributed scaling | Tesseract v1 targets single-node inference and does not implement distributed execution primitives. |
| Real-time latency guarantees | Diffusion-based 3D generation is inherently compute-intensive, and the system is not designed for interactive or real-time use cases. |
| Output quality or mesh fidelity evaluation | Output quality is determined by the underlying Shape-E model and is outside the scope of system-level performance evaluation. |
| Cloud cost optimization | Cost characteristics depend heavily on deployment environment and hardware pricing; only rough cost estimates are considered later. |
| High-concurrency load testing | The system prioritizes correctness and predictability of heavy inference workloads rather than high-QPS request handling. |


## Benchmark Environment

Benchmarks were executed on **Google Colab** with a single GPU assigned per session.  
All benchmark runs were conducted within the same runtime session to ensure hardware and software consistency across experiments.

### Hardware and System Configuration

| Component | Details |
|:--|:--|
| Execution Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU VRAM | 15,360 MiB |
| GPU Driver Version | 550.54.15 |
| CUDA Version (Driver) | 12.4 |
| CPU Architecture | x86_64 |
| CPU Model | Intel(R) Xeon(R) CPU @ 2.00 GHz |
| Physical Cores | 1 |
| Threads per Core | 2 |
| Logical CPUs | 2 |
| L3 Cache | 38.5 MiB |
| Virtualization | KVM (full) |
| System RAM | 12 GiB |
| Available RAM at Start | ~11 GiB |
| Swap | Disabled |
| NUMA Nodes | 1 |

### Software Environment

| Component | Version |
|:--|:--|
| Operating System | Ubuntu 22.04.4 LTS (jammy) |
| Python | 3.12.12 |
| PyTorch | 2.9.0+cu126 |
| Torch CUDA Runtime | 12.6 |

### GPU State at Benchmark Start

| Metric | Value |
|:--|:--|
| Persistence Mode | Off |
| Compute Mode | Default |
| Temperature | ~45 °C |
| GPU Utilization | 0% |

### Benchmarking Notes

- All GPU benchmarks were executed on the **same NVIDIA Tesla T4 instance**.
- CPU benchmarks were executed within the same Colab virtual machine.
- FP16 inference was enabled for GPU runs; CPU runs used default PyTorch precision.
- No concurrent GPU workloads were present during benchmarking.
- Reported results are averaged over **three runs** and presented as **mean ± sample standard deviation**.
- Deterministic seeding was not enforced; variance is captured via repeated runs.

## Results Overview

This section summarizes empirical findings from controlled benchmarking experiments.  
Detailed tables follow in subsequent sections.

At a high level, the benchmark results show:

- GPU execution outperforms CPU fallback by **multiple orders of magnitude** for diffusion-based 3D generation.
- Inference latency scales approximately linearly with batch size on GPU.
- Increasing sampling steps significantly increases latency while leaving memory usage largely unchanged.
- Prompt complexity has negligible impact on inference latency, indicating that performance is dominated by the diffusion process rather than prompt encoding.

## Results and Discussion

This section presents and interprets the benchmark results collected across
different execution devices, batch sizes, sampling configurations, and prompt
complexity levels. All results are reported as **mean ± sample standard deviation**
over three runs. Full aggregated tables are available under
`benchmarks/results/tables/`.

---

### GPU vs CPU Performance

The performance gap between GPU-accelerated execution and CPU fallback is
substantial under identical baseline configurations (batch size = 1, sampling
steps = 15, simple prompt). Table `gpu_vs_cpu_latency.md` summarizes these results.

GPU inference completes in **4.936 ± 0.029 s**, whereas CPU inference requires
**1635.15 ± 17.73 s** _(~27 mins)_, resulting in an approximate **330× slowdown** when falling
back to CPU execution. A similar disparity is observed for end-to-end latency,
with GPU execution completing in **6.069 ± 0.064 s** compared to
**1662.15 ± 16.50 s** _(~28 mins)_ on CPU.

These results demonstrate that CPU execution is functionally correct but
computationally impractical for diffusion-based 3D generation workloads. The
overhead introduced by CPU fallback dominates total execution time and strongly
motivates GPU availability for any realistic usage of the system.

---

### Inference vs End-to-End Latency Breakdown

Across all GPU runs, inference latency consistently accounts for the majority of
end-to-end latency. For the baseline configuration, inference constitutes
approximately **81–83%** of total execution time, with the remaining overhead
attributable to preprocessing, scheduling, and mesh decoding.

This separation validates the benchmarking approach and indicates that
optimization efforts aimed at reducing overall latency should primarily target
the inference pipeline rather than auxiliary system components.

---

### Batch Size Scaling on GPU

Batch size scaling behavior is summarized in `batch_size_scaling_gpu.md`. Increasing
batch size from 1 to 4 results in near-linear growth in both inference and
end-to-end latency.

- Batch size 1: **4.936 ± 0.029 s** inference latency
- Batch size 2: **10.129 ± 0.138 s**
- Batch size 4: **20.628 ± 0.823 s**

While absolute latency increases with batch size, this behavior enables higher
aggregate throughput at the cost of increased per-request latency.

---

### Effect of Sampling Steps

The impact of diffusion sampling steps is summarized in
`sampling_steps_scaling.md`. Increasing sampling steps from 15 to 64 leads to a
substantial increase in latency:

- Inference latency increases from **4.936 ± 0.029 s** to
  **23.627 ± 0.441 s**
- End-to-end latency increases from **6.069 ± 0.064 s** to
  **24.777 ± 0.368 s**

Despite the increased computational cost, peak GPU memory usage remains
unchanged at approximately **5.055 ± 0.002 GiB**, confirming that sampling depth
affects compute time but does not significantly alter memory footprint.

This result highlights sampling steps as a primary latency–quality tradeoff
parameter in the system.

---

### Prompt Complexity Analysis

Prompt complexity results are summarized in `prompt_complexity_gpu.md`. Inference
latency remains nearly invariant across simple, medium, and complex prompts:

- Simple: **4.936 ± 0.029 s**
- Medium: **4.921 ± 0.012 s**
- Complex: **4.960 ± 0.040 s**

End-to-end latency shows slightly higher variance with increasing prompt
complexity; however, these differences remain within a narrow range and do not
materially impact inference performance. GPU memory usage remains identical
across all prompt complexity levels.

These results indicate that prompt complexity has negligible impact on the core
diffusion inference workload, and minor variations in end-to-end latency are
likely attributable to preprocessing or system-level overhead rather than model
execution.

---
### GPU Memory Utilization

Peak GPU memory usage was measured for all GPU runs and reported as
**mean ± sample standard deviation**. Memory usage scales monotonically with batch
size, increasing from **5.055 ± 0.002 GiB** at batch size 1 to
**6.364 ± 0.009 GiB** at batch size 4.

Increasing sampling steps from 15 to 64 does not affect peak GPU memory usage,
which remains constant at approximately **5.055 ± 0.002 GiB**. Similarly, prompt
complexity has no measurable impact on GPU memory consumption.

Under the evaluated configurations, successful GPU execution requires **approximately
5.1 GiB of available GPU memory** at minimum. This value reflects the steady-state
memory footprint after model loading and warm-up, and does not include additional
headroom required by concurrent workloads or system overhead.

Across all configurations, peak GPU memory exhibits extremely low variance after
warm-up, indicating stable and deterministic memory allocation behavior during
inference.
---

### Failure Modes
- GPU OOM for large batch sizes
- CPU execution exceeding practical time limits
- Cold-start latency due to model loading
- Limited output quality due to upstream model

---

### Key Takeaways

The benchmark results support the following conclusions:

1. **GPU acceleration is essential** for practical execution of diffusion-based
   3D generation, with CPU fallback incurring orders-of-magnitude slowdown.
2. **Inference dominates end-to-end latency**, validating the focus on inference
   optimization for performance improvements.
3. **Batch size increases throughput at the cost of latency**, with predictable
   and stable memory scaling.
4. **Sampling steps are the dominant latency driver**, offering a clear
   quality–performance tradeoff.
5. **Prompt complexity has minimal impact on inference performance**, confirming
   that the diffusion process, rather than text processing, dominates runtime.
6. **GPU memory usage stabilizes after warm-up**, exhibiting low variance and
   deterministic behavior across runs.
7. **GPU memory footprint:** Tesseract requires ~**5.1 GiB** of GPU memory for  batch size 1 after warm-up. Memory usage scales with batch size and is invariant to sampling steps and prompt complexity.



Overall, these results provide a clear performance characterization of
Tesseract’s inference pipeline and establish a reliable baseline for future
optimization and system evolution.


### Minor Observations 
- After the warm-up run, peak GPU memory usage stabilized with negligible variance across repeated runs, indicating deterministic memory allocation behavior during inference.
- Inference latency remains nearly invariant across prompt complexity levels. Small variations observed are within run-to-run variance and are likely attributable to non-diffusion overhead such as preprocessing or system noise.
