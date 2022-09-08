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
