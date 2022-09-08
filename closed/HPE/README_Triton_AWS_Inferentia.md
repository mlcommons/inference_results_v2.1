# MLPerf Inference v2.0 NVIDIA-Optimized Implementations of Triton Inference Server running on AWS Inferentia
This is a repository of NVIDIA-optimized implementations for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
This README is a quickstart tutorial on how to use our code for Triton on AWS Inferentia systems as a public / external user.
It is recommended to also read README.md for general instructions


NVIDIA's Triton Inference Server is an open-source inference serving software for deploying trained AI models at scale in production, optimized for performance and designed to provide a consistent interface when running on GPUs or CPUs. In NVIDIA's MLPerf Inference v2.0 submission, Triton's Python Backend supports running pytorch and tensorflow models on Inferentia using the AWS Neuron SDK.


### NVIDIA Submissions

The Triton AWS Inferentia submission supports:

- BERT (Offline, Server), at 99% and 99.9% of FP32 accuracy target
- ResNet50 (Offline, Server), at 99% of FP32 accuracy target

See the main README.md for an explanation of the benchmarks and scenarios.

To generate the preprocessed datasets, follow the benchmark-specific instructions described in the `[README.md](http://README.md)` files stored in `code/[benchmark]/aws-neuron` for each benchmark.

### NVIDIA Submission Systems

The AWS Inferentia systems that NVIDIA supports, has tested, and is submitting to MLPerf Inference v2.0 are:

- AWS Inferentia inf1.2xLarge system on EC2
- AWS Inferentia inf1.6xLarge system on EC2

### Setup

#### Choosing an EC2 instance and logging in
Step 1: Choose the Deep Learning  Ubuntu 18.04 machine image
Step 2: Choose the Inferentia instance type you would like to use. NVIDIA has tested inf1.2xlarge and inf1.6xlarge instances.
Step 3: Customize storage to have at least
Step 4: Customing security settings according to organization . Use Key-Pair for login.
Step 5: Start the instance and login via SSH

#### Setting up MLPerf environment
To run the benchmarks on inferentia you will need to set up:
1. Create Inferentia optimized models using AWS Neuron
2. The MLPerf Inference code repository
3. The MLPerf scratch space with preprocessed datasets

##### Creating Inferentia optimized models for neuron
To run pytorch models optimized on inferentia, they have to be compiled using the torch neuron package. See more details here - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/index.html

For Resnet50, start with the torchvision resnet50 model, load the MLCommons state dict and re-compile the model for inference with dynamic_batch_size=true flag. The model is compiled for fp32 inputs and stored as:
     `/home/ubuntu/inferentia-compiled-models/resnet50/resnet50-pytorch-bs{batch_size}/resnet50_neuron_bs{batch_size}_dynamic.pt`

For Bert, start with the transformers Question Answer model, load the MLCommons pytorch model weights and concat the two outputs into a single output of size (384,2). The model is compiled for int32 outputs and stored as:
    `/home/ubuntu/inferentia-compiled-models/bert/bert-pytorch-bs{batch_size}/bert-large-int32-bs{batch_size}-concat.pt`

Additional details for how to compile the models for Inferentia can be found in AWS documentation - https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-guide/neuron-frameworks/pytorch-neuron/tutorials/index.html#pytorch-tutorials

##### The MLPerf Inference code repository

##### The MLPerf scratch space with preprocessed datasets
1. Create a directory at /home/ubuntu/mlperf_scratch with the following heirarchy
`/home/ubuntu/mlperf_scratch
|____preprocessed_data
    |__ squad_tokenized
    |__ imagenet
            |__Resnet50
                |__fp32_pytorch

|____data

`
2. Preprocess the imagenet data with code/resnet50/aws_neuron/preprocess_data.py and save it as numpy files at fp32_pytorch
3. Add the bert preprocessed Squad data to the squad_tokenized folder

### Setting up the containers and building the binaries

Like the GPU-based submission, `closed/NVIDIA` should be the working directory when running any commands. You should also **not execute any commands** with the user `root`. Doing so may cause permission errors.

We recommend using Ubuntu 18.04. Other operating systems have not been tested.

To use the Inferentia code paths, set the `USE_INFERENTIA` environment variable in order to properly build and set up the container.

```
export USE_INFERENTIA=1
export USE_NIGHTLY=0
make prebuild
```
The docker image will have the tag `mlperf-inference:[username]-latest`. The working directory (`closed/NVIDIA`) will also be mounted in `/work` inside the container. Once in the container, build the harness binaries with

```
make build_inferentia
```
This step will take some time as triton will build a conda environment with all the requirements to run the model and package it into the triton python backend stub.

### Running a Benchmark

To run a benchmark, use `make run_inferentia_harness`. Unlike the normal GPU-based NVIDIA submission, there is **no 'generate engines' step**, as that is a phase specific to TensorRT. In general, you can follow the instructions for the GPU-based submission in the main README.md, except use `run_inferentia_harness` as the make target, and set `--config_ver=triton in RUN_ARGS`.

As an example, to run ResNet50 in Offline scenario with the triton inferentia python backend, run:

```
make run_inferentia_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=triton"
```
To run in Accuracy mode, add `--test_mode=AccuracyOnly` to `RUN_ARGS`, just like with the GPU-based submission:

```
make run_inferentia_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=offline --config_ver=triton --test_mode=AccuracyOnly" VERBOSE=1
```
Replace `--benchmarks` and `--scenarios` as normal, just like with the GPU-based submission.

Follow the steps in the main README.md for instructions on compliance tests and making an actual submission.

