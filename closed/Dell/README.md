# MLPerf Inference v2.1 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v2.1](https://www.mlperf.org/inference-overview/).

# Implementations
## Benchmarks
**Please refer to /closed/NVIDIA for detailed instructions for NVIDIA GPU & Triton submissions, including performace guides, and instructions on how to run with new systems.** 

**Please refer to /closed/Qualcomm for detailed instructions for Qualcomm QAIC100 submissions, including performance guides, and instructions on how to run with new systems.**
  
The following benchmarks are part of our submission for MLPerf Inference v2.1:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrm](code/dlrm/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [retinanet](code/retinanet/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)

# Dell Technologies Submission Systems

The closed systems that Dell has submitted on are:
- Datacenter Systems
  - Dell PowerEdge R750xa
    - A100-PCIe-80GB
    - Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz
  - Dell PowerEdge R7515
    - QAIC100 Pro
  - Dell PowerEdge XE8545
    - A100-SXM-80GB / 500W
    - A100-SXM-80GB / 500W MaxQ
  - Dell PowerEdge XR12
    - A2 MaxQ
- Edge Systems
  - Dell PowerEdge R7515
    - QAIC100 Standard
  - Dell PowerEdge XR12
    - A2 MaxQ

