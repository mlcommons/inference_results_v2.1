# Getting started with Intel MLPerf Submission with Intel optimized Docker Images
This article guides users how to run Intel MLPerf Submission v2.1 with Intel Optimized Docker Images


## Intel Docker Images for MLPerf
This section guides users how to get Intel Optimized Docker Images for MLPerf Submission.

Here is the mapping table between model and its optimized docker image. 
|  Model  | Docker Image                               |
| --------------- | ------------------------------------ |
| 3dunet          | intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet                       |
| bert            | intel/intel-optimized-pytorch:mlperf-inference-2.1-bert                                  |
| dlrm            | intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm      |
| resnet50        | intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50                    |
| retinanet       | intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet |

Users could use below command to get related optimized docker image.
```
#dockerhub
docker pull <Docker Image>
```

## HW configuration:

| **System Info** | config                               |
| --------------- | ------------------------------------ |
| CPU             | E3 Q175 2S 56C                       |
| BKC             | #63                                  |
| BIOS            | E5C6301.86B.7314.D09.2202231344      |
| OS              | CentOS  Stream 8                     |
| Kernel          | Linux 5.15.0-spr.bkc.pc.7.7.4.x86_64 |
| Microcode       | 0xab000060                           |
| memory          | 1024GB (16x64GB 4800MT/s [4800MT/s]) |
| disk            | 1TB nvme                             |

Best Known Configurations:

```
echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo 0 > /proc/sys/kernel/numa_balancing
cpupower frequency-set -g performance
```




## Getting started with 3DUNET

### Prerequisites
This session guide users how to prepare dataset and model on host

```
mkdir 3dunet
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
```

### Environment Setup
This session guide users how to setup the docker instance and preprocess data 

#### start a container
Please replace </path/of/3dunet/above> with the 3dunet folder path from Prerequisites session.
```
docker run --name 3dunet_2-1 --privileged -itd --net=host --ipc=host -v </path/of/3dunet/above>:/data/mlperf_data/ intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet
```

#### into docker container
Login into a bashrc shell in the docoker instance.
```
docker exec -it 3dunet_2-1 bash
```

#### preprocess data
Please replace "your host proxy" with the proxy server for your environment.
Ignore the http_proxy and https_proxy if you are not under a proxy server.
```
cd code/3d-unet-99.9/pytorch-cpu-kits19
export http_proxy="your host proxy"
export https_proxy="your host proxy"
export DOWNLOAD_DATA_DIR=/data/mlperf_data/3dunet/kits19/data
bash make_preprocess.sh
```

### How to Run the Benchmark

```
# 3dunet only have offline mode
bash run_SPR56C_2S.sh   # offline performance
bash run_SPR56_2S.sh acc  # offline accuracy
```

#### parameter tuning

in run_SPR56C_2S.sh: 	

```
--num-instance: the number of instances, num-instance*cpus-per-instance=physical_cores. 
--cpus-per-instance: the number of cores bound to each instance, Please avoid binding cores across numa nodes.
#recommend cpus-per-instance=4, and set num-instance=core_number/cpus-per-instance. you also can tune it to achieve better performance.
```

in user.conf:

```
*.Offline.target_qps: Adjust according to the performance
# You can make adjustments based on reference test results, scalling it by 
ref_QPS *(physical_cores/physical_cores_of_ref), you can increase it if the performance is better.
# if you want to save time for tuning performance, please use a small amount of data by:
*.Offline.min_query_count = 1200
*.Offline.min_duration = 6000
```

### How to get the results

please check log file, performance in ./output/mlperf_log_summary.txt, make sure the "results is: valid" 

```
for offline mode performance, please check the field “Samples per second: ”
```

 accuracy in ./output/accuracy.txt

```
please check the field "mean = "
```

Please save it when each test is completed or it will be overwritten by the next test.



##  Getting started with BERT

### Prerequisites
This session guide users how to prepare dataset and model on host

```
mkdir bert
cd bert
mkdir dataset
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dataset/dev-v1.1.json
git clone https://huggingface.co/bert-large-uncased model
cd model
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
```

### Environment Setup
This session guide users how to setup the docker instance and preprocess data 

#### start a container
Please replace </path/of/bert/above> with the 3dunet folder path from Prerequisites session.
```
docker run --name bert_2-1 --privileged -itd --net=host --ipc=host -v </path/of/bert/above>:/data/mlperf_data/bert intel/intel-optimized-pytorch:mlperf-inference-2.1-bert
```

#### into docker container
Login into a bashrc shell in the docoker instance.
```
docker exec -it bert_2-1 bash
```

#### convert dataset and model

```
cd bert-99/pytorch-cpu
export DATA_PATH=/data/mlperf_data/bert
bash convert.sh
```

### How to Run the Benchmark

```
bash run.sh    #offline performance
bash run.sh --accuracy    #offline accuracy
bash run_server.sh   #server performance
bash run_server.sh --accuracy    #server accuracy
```

#### parameter tuning

in run.sh(offline) or run_server.sh (server):

```
-n, --inter_parallel: [number] Instance Number, n*j=physical_cores
-j, --intra_parallel: [number] Thread Number Per-Instance, Please avoid binding cores across numa nodes
#recommend -j 4, n=physical_cores/j, you also can tune it to achieve better performance.
```

in user.conf:

```
bert.Offline.target_qps: Adjust according to the performance
bert.Server.target_qps: Adjust according to the performance
# You can make adjustments based on reference test results, scalling it by 
ref_QPS *(core_number/core_number_of_ref), you can increase it if the performance is better.
# for server mode, under the premise of ensuring that the result is valid, increase the target_qps as much as possible to make "99.00 percentile latency" close to "target_latency". If the "99.00 percentile latency" exceed the "target_latency", the results will be invalid."result is", "99.00 percentile latency" and "target_latency" both in mlperf_log_summary.txt
```

### How to get the results

please check log file, performance in ./test_log/mlperf_log_summary.txt, make sure the "results is: valid" 

```
 for offline mode performance, please check the field "Samples per second: "
 for server mode performance, please check the field "Scheduled samples per second: "
```

 accuracy in ./test_log/accuracy.txt. 

```
please check the field "f1"
```

Please save it when each test is completed.



## Getting started with DLRM

### Prerequisites
This session guide users how to prepare dataset and model on host

```
mkdir dlrm
cd dlrm
#dataset contain:
#     day_fea_count.npz
#     terabyte_processed_test.bin
#   About how to get the dataset, please refer to
#      https://github.com/facebookresearch/dlrm
#you can also copy it by: scp -r mlperf@10.112.230.156:/home/mlperf/dlrm_data/* dlrm/
#model
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### Environment Setup
This session guide users how to setup the docker instance and preprocess data 
#### start a container
Please replace </path/of/dlrm/above> with the 3dunet folder path from Prerequisites session.
```
for DLRM, please run it with HT off
docker run --name dlrm_2-1 --privileged -itd --net=host --ipc=host -v </path/of/dlrm/above>:/data/mlperf_data/dlrm intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm
```

#### into docker container
Login into a bashrc shell in the docoker instance.
```
docker exec -it dlrm_2-1 bash
```

### How to Run the Benchmark

```
cd code/dlrm-99.9/pytorch-cpu/
bash run_mlperf.sh --mode=offline --type=perf --dtype=int8
#mode: offline or server
#type: perf or acc
```

#### parameter tuning

in setup_env_offline.sh(offline) or setup_env_server.sh (server):

```
CPUS_PER_SOCKET: adjust it according to your system
CPUS_PER_PROCESS: which determine how much processes will be used
                  process-per-socket = CPUS_PER_SOCKET/CPUS_PER_PROCESS
CPUS_PER_INSTANCE: instance-per-process number=CPUS_PER_PROCESS/CPUS_PER_INSTANCE
                   total-instance = instance-per-process * process-per-socket
BATCH_SIZE: batch size, you can try to tune it to get better performance.
#recommend CPUS_PER_INSTANCE=1, CPUS_PER_PROCESS=CPUS_PER_SOCKET
```

in user.conf:

```
dlrm.Offline.target_qps: Adjust according to the performance
dlrm.Server.target_qps: Adjust according to the performance
# You can make adjustments based on reference test results, scalling it by 
ref_QPS *(core_number/core_number_of_ref), you can increase it if the performance is better.
# for server mode, under the premise of ensuring that the result is valid, increase the target_qps as much as possible to make "99.00 percentile latency" close to "target_latency".If the "99.00 percentile latency" exceed the "target_latency", the results will be invalid. "result is", "99.00 percentile latency" and "target_latency" both in mlperf_log_summary.txt
```

### How to get the results

please check log file, performance in ./output/pytorch-cpu/dlrm/<Offline,server>/performance/run_1/mlperf_log_summary.txt, make sure the "results is: valid" .

```
 for offline mode performance, please check the field "Samples per second: "
 for server mode performance, please check the field "Scheduled samples per second: "
```

 accuracy in ./output/pytorch-cpu/dlrm/<Offline,server>/accuracy/accuracy.txt. 

```
please check the field "roc_auc"
```

Please save it when each test is completed.



##  Getting started with ResNet50

### Prerequisites
This session guide users how to prepare dataset and model on host
```
# ImageNet(50000) datatset
bash download_imagenet.sh
#prepare calibration 500 images into folders
bash prepare_calibration_dataset.sh
#model
bash download_model.sh
```
### Environment Setup
This session guide users how to setup the docker instance and preprocess data 
#### start a container

```
docker run --name resnet50_2-1 --privileged -itd --net=host --ipc=host intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50
```

#### into docker container
Login into a bashrc shell in the docoker instance.
Please replace "your host proxy" with the proxy server for your environment.
Ignore the http_proxy and https_proxy if you are not under a proxy server.
```
docker exec -it resnet50_2-1 bash
cd code/resnet50/pytorch-cpu
export http_proxy=<http_proxy>
export https_proxy=<https_proxy>
```


#### Quantize Torchscript Model and Check Accuracy

```
#set path
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth

#Generate scales and models
bash generate_torch_model.sh

##The start and end parts of the model are also saved (respectively named) in models
```

### How to Run the Benchmark

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth

#offline performance
bash run_offline.sh
#sever performance
bash run_server.sh

#accuracy
#offline
bash run_offline_accuracy.sh
python -u ./mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
--mlperf-accuracy-file mlperf_log_accuracy.json \
--imagenet-val-file ${DATA_DIR}/val_map.txt \
--dtype int32 2>&1 | tee accuracy.txt
#server
bash run_server_accuracy.sh
python -u ./mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
--mlperf-accuracy-file mlperf_log_accuracy.json \
--imagenet-val-file ${DATA_DIR}/val_map.txt \
--dtype int32 2>&1 | tee accuracy.txt
```

#### parameter tuning

in run_<offline/server>.sh /run_<offline/server>_accuracy.sh file:

```
--num_instance 224 \     # please set it according to you HW, for offline, according Logical cores, for server, =physical_cores/cpus_per_instance
--warmup_iters 20 \
--cpus_per_instance 1\     #for offline, recommend set to 1, for server, recommend set to 4
--total_sample_count 50000 \
--batch_size 9

```

### How to get the results

please check log file, performance in ./mlperf_log_summary.txt, make sure the "results is: valid" 

```
 for offline mode performance, please check the field "Samples per second: "
 for server mode performance, please check the field "Scheduled samples per second: "
```

 accuracy in ./accuracy.txt. 

```
please check the field "accuracy"
```

Please save it when each test is completed.



##  Getting started with Retinanet

### Prerequisites
This session guide users how to prepare dataset and model on host
```
# download openimages(264) dataset
export WORKLOAD_DATA=${PWD}/data
mkdir -p $WORKLOAD_DATA
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages
#Download Calibration images
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration
#download model
wget --no-check-certificate 'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```
### Environment Setup
This session guide users how to setup the docker instance and preprocess data 

#### start a container

```
docker run --name retinanet_2-1 --privileged -itd --net=host --ipc=host intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet
```

#### into docker container
Login into a bashrc shell in the docoker instance.
Please replace "your host proxy" with the proxy server for your environment.
Ignore the http_proxy and https_proxy if you are not under a proxy server.
```
docker exec -it retinanet_2-1 bash
cd code/retinanet/pytorch-cpu
export http_proxy=<http_proxy>
export https_proxy=<https_proxy>
```


#### calibrate and generate torchscript model

```
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

### How to Run the Benchmark

```
export DATA_DIR=${WORKLOAD_DATA}/openimages
export MODEL_PATH=${WORKLOAD_DATA}/retinanet-int8-model.pth

#offline performance
bash run_offline.sh
#sever performance
bash run_server.sh

##accuracy
#offline accuracy
bash run_offline_accuracy.sh
#server accuracy
bash run_server_accuracy.sh
```

#### parameter tuning

in run_<offline/server>.sh /run_<offline/server>_accuracy.sh file:

```
--cpus_per_instance #for offline, recommend set to 4, for server, recommend set to 8
--num_instance # please set it according to you HW, =physical_cores/cpus_per_instance
--batch_size 2
```
### How to get the results

please check log file, performance in ./mlperf_log_summary.txt, make sure the "results is: valid" 

```
 for offline mode performance, please check the field "Samples per second: "
 for server mode performance, please check the field "Scheduled samples per second: "
```

 accuracy in ./accuracy.txt. 

```
please check the field "mAP"
```

Please save it when each test is completed.
