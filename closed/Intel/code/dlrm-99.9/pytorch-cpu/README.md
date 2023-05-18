# Setup from Source

## HW and SW requirements
### 1. HW requirements
| HW  |      Configuration      |
| --  | ----------------------- |
| CPU | SPR-6 @ 2 sockets/Node  |
| DDR | 512G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T      |

### 2. SW requirements
| SW       | Version |
|----------|---------|
| GCC      |  11.2   |
| Binutils | >= 2.35 |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
  mkdir <workfolder>
  cd <workfolder>
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ./anaconda3.sh -b -p ./anaconda3
  export WORKDIR=$PWD
  export PATH=$WORKDIR/anaconda3/bin:$PATH
  conda create -n dlrm python=3.9
  conda update -n base -c defaults conda --yes
  conda activate dlrm
```
### 2. Download Repo for DLRM MLPerf inference
```
  git clone <path/to/this/repo> 
  ln -s <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch-cpu dlrm_pytorch
```
### 3. Install conda dependency packages
```
  cp dlrm_pytorch/prepare_conda_env.sh .
  bash ./prepare_conda_env.sh
```
### 4. Prepare GCC11
```
  If system has GCC11 installed, you can try to update GCC version by using:
    source /opt/rh/gcc-toolset-11/enable
  Or you can install GCC11.2 through conda by using following scripts:
    source ./setup_gcc_env.sh #install and setup GCC11 env
```
### 5. Install mlperf loadgen and intel extension for pytorch
```
  cp dlrm_pytorch/prepare_env.sh .
  bash prepare_env.sh
```
### 6. Prepare DLRM dataset and code
(1) Prepare DLRM dataset
```
   Create a directory (such as /data/mlperf_data/dlrm/) which contain:
     day_fea_count.npz
     terabyte_processed_test.bin

   About how to get the dataset, please refer to
      https://github.com/facebookresearch/dlrm
```
(2) Prepare pre-trained DLRM model
```
   cd /data/mlperf_data/dlrm/
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
### 7. Run command for server and offline mode

(1) cd dlrm_pytorch

(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
```
   export DATA_DIR=           # the path of dataset, for example as /data/mlperf_data/dlrm/
   export MODEL_DIR=          # the path of pre-trained model, for example as /data/mlperf_data/dlrm/
```
(3) configure offline/server mode options # currenlty used options for each mode is in setup_env_xx.sh, You can modify it, then 'source ./setup_env_xx.sh'
```
   export NUM_SOCKETS=        # i.e. 8
   export CPUS_PER_SOCKET=    # i.e. 28
   export CPUS_PER_PROCESS=   # i.e. 14. which determine how many cores for one processe running on one socket
                              #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
   export CPUS_PER_INSTANCE=  # i.e. 14. which determine how many cores used for one instance inside one process
                              #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                              #   total_instance_number_in_system = instance_number_per_process * process_number
```
(5) Disable Hyper-threading and set system to performance mode
   echo off  > /sys/devices/system/cpu/smt/control  # disable hyper-threading
   sudo ./set_perf.sh           # set system to performance mode
```
(6) command line
   Please update setup_env_server.sh and setup_env_offline.sh and user.conf according to your platform resource.
   bash run_mlperf.sh --mode=<offline/server> --type=<perf/acc> --dtype=int8
```
   for int8 calibration scripts, please look into Intel/calibration/dlrm/pytorch-cpu/ directory.
   for int8 execution, calibration result is in int8_configure.json which is copied from that output.



# Setup with Docker

###  1. Prepare dataset and model in host

```
please follow the step above to prepare the dataset and model on host system.
```
###  2. Start and login to a container

```
# -v mount the dataset to docker container 
docker run --privileged --name intel_dlrm -itd --net=host --ipc=host -v </path/to/dataset/and/model>:/data/mlperf_data/dlrm intel/intel-optimized-pytorch:mlperf-submission-inference-2.1-dlrm99.9

#check container, it will show a container named intel_dlrm
docker ps

# into container
docker exec -it intel_dlrm bash
```
###  3. Run DLRM

```
cd /opt/workdir/code/dlrm-99.9/pytorch-cpu 
```
```
please follow the #step7- Run command for server and offline mode above
```


# Run on host with docker image (automation script)
You could run the workload with prepared docker image, without going into the docker image container. This script can automatically process running docker containers with minimal user intervention, using the pre-trained models and datasets that are saved outside of the container environment.
###  1. Prepare dataset and model in host
```
please follow the step above to prepare the dataset and model on host system.
```
###  2. Run DLRM
+ Offline performance
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm   /workspace/run_offline.sh  /opt/workdir/code/dlrm-99.9/pytorch-cpu aws_dlrm  dlrm_perf.txt
```
+ Offline accuracy
```
../../run_docker.sh intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm   /workspace/run_offline_accuracy.sh  /opt/workdir/code/dlrm-99.9/pytorch-cpu aws_dlrm  dlrm_acc.txt
```
