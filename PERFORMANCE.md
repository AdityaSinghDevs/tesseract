## Performance Goals

The performance goals of Tesseract are focused on understanding and characterizing system behavior rather than minimizing raw latency.

The primary goals are:

1. End-to-end latency transparency across the request lifecycle.
2. Performance quantification between GPU-accelerated and CPU-fallback execution.
3. Effect of configuration parameters and batch size on latency and throughput.
4. Memory usage patterns during inference, particularly peak GPU memory consumption.
5. Making the tradeoffs between portability and performance explicit.

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
| Memory pressure | Peak GPU memory | Identifies memory bottlenecks and OOM risk|

## Test Matrix

Benchmarks are executed across a controlled set of configurations to ensure fair comparison and avoid cherry-picking results.

| Fixed parameters | Variable parameters  |
|:-----------------|:---------------------|
| Prompt(s):fixed set of representative prompts with varying complexity | *Device:* GPU, CPU |
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

Benchmarks were executed on Google Colab with a single GPU assigned per session. All benchmark runs were conducted within the same runtime session to ensure hardware consistency.

| Component | Details |
|:----------|:--------|
| Execution platform | Google Colab |
| GPU | NVIDIA RTX 3060 (12 GB VRAM) |
| CPU | Intel Core i7-12700H |
| System RAM | 32 GB |
| Operating System | Ubuntu 22.04 LTS |
| CUDA Version | 11.8 |
| PyTorch Version | 2.x |
| Python Version | 3.10 |

