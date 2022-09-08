# MLPerf Inference v2.1 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v2.1](https://www.mlperf.org/inference-overview/).

# Implementations
## Benchmarks
These submissions are in open division, but in many cases use the same instructions & implementations as closed division with some changes. 

**Please refer to /closed/NVIDIA for detailed instructions for NVIDIA GPU & Triton submissions, including performace guides, and instructions on how to run with new systems.**

**Please refer to /closed/Qualcomm for detailed instructions for Qualcomm QAIC100 submissions, including performance guides, and instructions on how to run with new systems.**

The following benchmarks are part of our open submission for MLPerf Inference v2.1:
- [bert](code/bert/tensorrt/README.md)
- [retinanet](code/retinanet/README.md)

# Dell Technologies Submission Systems

The open systems that Dell has submitted on are:
- Datacenter Systems
  - Dell PowerEdge R750xa
    - Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz
  - Dell PowerEdge R7525
    - AMD EPYC 7773X 64-Core Processor
  - Dell PowerEdge R7515
    - QAIC100 Pro
- Edge Systems
  - Dell PowerEdge R7515
    - QAIC100 Standard


