# MLPerf Inference v2.1 Implementations

This is a repository of the `Moffett` using optimized implementations
for [MLPerf Inference Benchmark v2.1](https://www.mlperf.org/inference-overview/).

## Moffett Submission Systems  

Moffett is dedicated to providing AI computing platforms and services, with a mission to keep evolving the frontiers of AI using sparse computing. 
Moffett submits multiple systems for datacenter inference category with three types of PCIe AI Accelerator Card. 

* PCIe Half-Height Half-Length (HHHL) SparseOne® S4 card
* PCIe Full-Height Full-Length (FHFL) SparseOne® S10 card 
* PCIe Full-Height Full-Length (FHFL) SparseOne® S30 card 

The SparseOne cards are based on Moffett's 1st generation AI silicon chip, Antoum® chip which natively supports up to 32x weight sparsity models. 
The systems that Moffett has submitted using Moffett SparseOne® AI Accelerator cards are:
- Datacenter systems
     - Inspur NF5280M6
        - CPU: 2 * Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
        - RAM: 16 * 32GB 3200Mhz Total:512GB
        - Disk: 1 * 1.92TB SATA SSD
        - **Accelerator: SparseOne® S4-PCIe/HHHL-20GB**
     - Dell R750
        - CPU: 2 * Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
        - RAM: 8 * 32GB 3200Mhz Total:512GB
        - Disk: 1 * 1.92TB SATA SSD
        - **Accelerator: SparseOne® S10-PCIe/FHFL-40GB** 
     - Dell R750
        - CPU: 2 * Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
        - RAM: 8 * 32GB 3200Mhz Total:512GB
        - Disk: 1 * 1.92TB SATA SSD
        - **Accelerator: SparseOne® S30-PCIe/FHFL-60GB**

## Benchmarks

The following benchmarks are part of our submission for MLPerf Inference v2.1:

* resnet50
* bert-99.9

## Scenarios

The above benchmarks can run in the following inference scenarios:

* Offline

Please refer to
the [MLPerf Inference official page](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#3-scenarios)
for explanations about the scenarios.

# Moffett Submission

Our MLPerf Inference v2.1 implementation has the following submissions:

| Benchmark | Datacenter Submissions |
| :-------- | :--------------------- |
| resnet50  | 99% of FP32 accuracy, Offline   |
| bert-99.9  | 99.9% of FP32 accuracy, Offline   |

The benchmark is stored in the `code/` directory which contains a `README.md` detailing the instructions on how to set
up the benchmark.
