## Setup Instructions

### Setup Conda Environment and Build binaries
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
+ Setup conda environment for to install requirements, and build the src code
```
CUR_DIR=$(pwd)
git clone <path/to/this/repo>
cd <path/to/this/repo>/closed/Intel/code/retinanet/pytorch-cpu
bash prepare_env.sh
conda activate retinanet-env
```

### Setup with docker image

you can skip the steps of setup conda env and build binaries, follow the steps to use docker image.

```
# start a docker container and login
docker run --name intel_retinanet --privileged -itd --net=host --ipc=host intel/intel-optimized-pytorch:mlperf-submission-inference-2.1-retinanet

docker exec -it intel_retinanet bash 
cd retinanet/pytorch-cpu/
export http_proxy=<your/proxy>
export https_proxy=<your/proxy>
```

### Download the dataset

+ Setup env vars
```


CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}

export ENV_DEPS_DIR=${CUR_DIR}/retinanet-env
```

+ Download OpenImages (264) dataset
```
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
```
Images are downloaded to `${WORKLOAD_DATA}/openimages`

+ Download Calibration images
```
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
```
Calibration dataset downloaded to `${WORKLOAD_DATA}/openimages-calibration`


### Download Model
```
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```

### Calibrate and generate torchscript model

Run Calibration
```
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

### Run Benchmark

```
export DATA_DIR=${WORKLOAD_DATA}/openimages
export MODEL_PATH=${WORKLOAD_DATA}/retinanet-int8-model.pth
```
#### Performance
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

### Automation scripts with docker image
You could run the workload with prepared docker image, without going into the docker image container. This script can automatically process running docker containers with minimal user intervention, using the pre-trained models and datasets that are saved outside of the container environment. 
+ Download Dataset and models. The dataset and models are saved in your environment, rather than the docker container environment.
Download OpenImages (264) dataset
```
./run_docker_download_images.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/openimages_mlperf.sh /opt/workdir/code/retinanet/pytorch-cpu aws_ret
```
Download Calibration images
```
./run_docker_download_images.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/openimages_calibration_mlperf.sh /opt/workdir/code/retinanet/pytorch-cpu aws_ret
```
Download Model
```
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ./data
```

+ Run the automation script 

Offline Performance 
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/run_offline.sh  /opt/workdir/code/retinanet/pytorch-cpu aws_ret  retinanet_offline_perf.txt
```
Offline Accuracy
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/run_offline_accuracy.sh  /opt/workdir/code/retinanet/pytorch-cpu aws_ret  retinanet_offline_acc.txt
```
Server Performance 
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/run_server.sh  /opt/workdir/code/retinanet/pytorch-cpu aws_ret  retinanet_server_perf.txt
```
Server Accuracy
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet /workspace/run_server_accuracy.sh  /opt/workdir/code/retinanet/pytorch-cpu aws_ret  retinanet_server_acc.txt
```
