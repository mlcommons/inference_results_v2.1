## MLPerf Inference v2.1 - Calibration
We have three types of calibration for the MLPerf submissions.

### 1. Alibaba Cloud Server Sinian Platform (vODLA-EFLO) systems - Calibration

For Alibaba Sinian vODLA-EFLO systems, we use the same calibration method as NVIDIA does. Please refer to NVIDIA's calibration document for more details.

### 2. Alibaba Cloud Server Sinian Platform (Panjiu-M) systems - Calibration
##### SinianML Quantization Aware Training (QAT)
SinianML QAT employs per-channel symmetric quantization for weight tensors and per-tensor asymmetric quantization for activation tensors.

##### Activation
`Per-tensor asymmetric` quantization is used. Collecting activation the min/max value with validation datasets listed in MLPerf. Next, train quantization parameters through SinianML QAT. Activation tensors were quantized to `int8`.

##### Weights
`Per-channel symmetric` quantization is used. First, find the maximum absolute value of each output channel of the weight tensors. Second, train the quantized model with SinianML QAT method in training datasets listed in MLPerf. Weight tensors were quantized to `int8` and bias tensors were quantized to `int32`.

##### Additional Details
We use Pytorch>=1.9.0 and SinianML QAT for quantization.

##### Reference
https://pytorch.org/docs/stable/quantization.html
  
### 3. Alibaba Cloud Elastic Compute Service GPU Cluster with ApasraCompute AIACC Tool - Calibration
For the Alibaba Cloud Elastic Compute Service GPU Cluster with ApasraCompute AIACC Tool, we use the same calibration method as NVIDIA does. Please refer to (the) NVIDIA's calibration document for more details.

