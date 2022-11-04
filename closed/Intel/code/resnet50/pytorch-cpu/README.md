## Setup Instructions

### REQUIRMENTS:
+ GCC > 10.1
+ NUMACTL
    + CentOS:
    ```
    sudo yum install numactl-devel
    sudo yum install numactl
    ```
    
    + Ubuntu:
    ```
    sudo apt install numactl-dev
    ```

### Download and sync submodules
+ From </path/to/repo/root>/closed/Intel/code/resnet50/pytorch-cpu
```
git submodule sync
git submodule update --init --recursive
```
+ From </path/to/repo/root>/closed/Intel/code/resnet50/pytorch-cpu/src/ckernels
```
mkdir 3rdparty
cd 3rdparty
git clone -b v2.6 https://github.com/oneapi-src/oneDNN.git onednn
```

### Setup Conda Environment
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
  

+ Setup conda environment to install requirements, and build packages
  ```
  bash prepare_env.sh
  ```
###Setup with docker image
  you can skip the setup conda env and use a docker image
  ```
  docker run --name intel_resnet50 --privileged -itd --net=host --ipc=host intel/intel-optimized-pytorch:mlperf-submission-inference-2.1-resnet50
  docker exec -it intel_resnet50 bash
  cd code/resnet50/pytorch-cpu
  export http_proxy=<your/proxy>
  export https_proxy=<your/proxy>
  ```


### Download Imagenet Dataset for Calibration
Download ImageNet (50000) dataset
```
bash download_imagenet.sh
```
Prepare calibration 500 images into folders 
```
bash prepare_calibration_dataset.sh
```

### Download Model
```
bash download_model.sh
```
The downloaded model will be saved as ```resnet50-fp32-model.pth```

### Quantize Torchscript Model and Check Accuracy 
+ Set the following paths:
```
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth
```
+ Generate scales and models
```
bash generate_torch_model.sh
```

The *start* and *end* parts of the model are also saved (respectively named) in ```models```

### Build 
```
bash build_binaries.sh
```

### Run Benchmark

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth
```

### Performance
+ Offline
```
run_offline.sh
```

+ Server
```
run_server.sh
```

#### Accuracy
+ Offline
```
run_offline_accuracy.sh
```

+ Server
```
run_server_accuracy.sh
```

### Run on host with docker image (automation script)
You could run the workload with prepared docker image, without going into the docker image container. This script can automatically process running docker containers with minimal user intervention, using the pre-trained models and datasets that are saved outside of the container environment.
+ Download Imagenet Dataset and models following the above steps. The dataset and models are saved in your environment, rather than the docker container environment.
```
bash download_imagenet.sh
bash prepare_calibration_dataset.sh
bash download_model.sh
```
+ Run the automation script 
Offline Performance
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50 /workspace/run_offline.sh  /opt/workdir/code/resnet50/pytorch-cpu aws_rn50  resnet50_offline_perf.txt
```
Offline Accuracy
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50 /workspace/run_offline_accuracy.sh  /opt/workdir/code/resnet50/pytorch-cpu aws_rn50  resnet50_offline_acc.txt
```
Server Performance
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50 /workspace/run_server.sh  /opt/workdir/code/resnet50/pytorch-cpu aws_rn50  resnet50_server_perf.txt
```
Server Accuracy
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50 /workspace/run_server_accuracy.sh  /opt/workdir/code/resnet50/pytorch-cpu aws_rn50  resnet50_server_acc.txt
```
