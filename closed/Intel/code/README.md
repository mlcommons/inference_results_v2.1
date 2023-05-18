# Get Started with Intel MLPerf v2.1 Submission with Intel Optimized Docker Images

MLPerf\* is a benchmark for measuring the performance of machine learning

systems. It provides a set of performance metrics for a variety of machine
learning tasks, including image classification, object detection, machine
translation, and others. The benchmark is representative of real-world
workloads and as a fair and useful way to compare the performance of different
machine learning systems.

Find out more information about the MLPerf v2.1 benchmark at
https://mlcommons.org/en/inference-datacenter-21/  and
https://mlcommons.org/en/inference-edge-21/.

In this document, we'll show how to run Intel MLPerf v2.1 submission with Intel
optimized Docker images.

## Intel Docker Images for MLPerf

Retrieve the Intel optimized Docker image for MLPerf v2.1 using a
``docker pull image_name`` command and specifying the corresponding model's
image name, as shown in the following table:

|  Model  | Intel optimized Docker image name            |
| --------------- | ------------------------------------ |
| 3dunet          | intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet                       |
| bert            | intel/intel-optimized-pytorch:mlperf-inference-2.1-bert                                  |
| dlrm            | intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm      |
| resnet50        | intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50                    |
| retinanet       | intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet |

For example:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet
```

## HW configuration:

| System Info     | Configuration detail                 |
| --------------- | ------------------------------------ |
| CPU             | E3 Q175 2S 56C                       |
| BKC             | #63                                  |
| BIOS            | E5C6301.86B.7314.D09.2202231344      |
| OS              | CentOS  Stream 8                     |
| Kernel          | Linux 5.15.0-spr.bkc.pc.7.7.4.x86_64 |
| Microcode       | 0xab000060                           |
| Memory          | 1024GB (16x64GB 4800MT/s [4800MT/s]) |
| Disk            | 1TB NVMe                             |

Best Known Configurations:

```
echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct
echo 0 > /proc/sys/kernel/numa_balancing
cpupower frequency-set -g performance
```

In the following sections, we'll show you how to set up and run each of the five models:

* [3DUNET](#get-started-with-3dunet)
* [BERT](#get-started-with-bert)
* [DLRM](#get-started-with-dlrm)
* [RESNET50](#get-started-with-resnet50)
* [RETINANET](#get-started-with-retinanet)

---

## Get Started with 3DUNET
If you haven't already done so, pull the Intel optimized Docker image for 3DUNET using:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet
```

### Prerequisites
Use these commands to prepare the 3DUNET dataset and model on your host system:

```
mkdir 3dunet
cd 3dunet
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
```

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/3dunet`` with the 3dunet folder path created earlier:
```
docker run --name 3dunet_2-1 --privileged -itd --net=host --ipc=host \
  -v /path/of/3dunet:/data/mlperf_data/ intel/intel-optimized-pytorch:mlperf-inference-2.1-3dunet
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it 3dunet_2-1 bash
```

#### Preprocess Data
If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment.  If no proxy is needed, you can skip
this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```

Use these commands to preprocess the data using the provided script:
```
cd code/3d-unet-99.9/pytorch-cpu-kits19
export DOWNLOAD_DATA_DIR=/data/mlperf_data/3dunet/kits19/data
bash make_preprocess.sh
```

### Run the Benchmark

```
# 3dunet only has offline mode
bash run_SPR56C_2S.sh     # offline performance
bash run_SPR56_2S.sh acc  # offline accuracy
```

#### Tune Parameters

In the ``run_SPR56C_2S.sh`` script, you can tune these parameters: 	

* ``--num-instance``: the number of instances

  Recommended value is ``number of cores / cpus-per-instance``.
* ``--cpus-per-instance``: the number of cores bound to each instance

   Avoid binding cores across numa nodes. Recommended value is 4.


In ``user.conf`` you can tune these parameters:

* ``*.Offline.target_qps``: Adjust according to the performance

  You can make adjustments based on reference test results, scaling it by
  ``ref_QPS * (physical_cores/physical_cores_of_ref)``. Increase the value
  if the performance is better.

* If you want to save time for tuning performance, use a small amount of data by adjusting:

  ```
  *.Offline.min_query_count = 1200
  *.Offline.min_duration = 6000
  ```

### Get the Results

* Check log file. Performance results are in ``./output/mlperf_log_summary.txt``.
  Verify that you see ``results is: valid``.

* For offline mode performance, check the field ``Samples per second:``
* Accuracy results are in ``./output/accuracy.txt``.  Check the field ``mean =``.

Save these output log files elsewhere when each test is completed as
they will be overwritten by the next test.

##  Get started with BERT
If you haven't already done so, pull the Intel optimized Docker image for BERT using:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-bert
```
### Prerequisites
Use these commands to prepare the BERT dataset and model on your host system:

```
mkdir bert
cd bert
mkdir dataset
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O dataset/dev-v1.1.json
git clone https://huggingface.co/bert-large-uncased model
cd model
wget https://zenodo.org/record/4792496/files/pytorch_model.bin?download=1 -O pytorch_model.bin
```

### Set Up Environment
Follow these steps to set up the docker instance and preprocess the data.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace /path/of/bert with the bert folder path created earlier:

```
docker run --name bert_2-1 --privileged -itd --net=host --ipc=host \
  -v /path/of/bert:/data/mlperf_data/bert intel/intel-optimized-pytorch:mlperf-inference-2.1-bert
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.
```
docker exec -it bert_2-1 bash
```

#### Convert Dataset and Model

```
cd bert-99/pytorch-cpu
export DATA_PATH=/data/mlperf_data/bert
bash convert.sh
```

### Run the Benchmark

```
bash run.sh                    #offline performance
bash run.sh --accuracy         #offline accuracy
bash run_server.sh             #server performance
bash run_server.sh --accuracy  #server accuracy
```

#### Tune Parameters

In ``run.sh`` (offline) or ``run_server.sh`` (server) scripts,
you can tune these parameters:

* ``-n``, ``--inter_parallel``: [number] Instance Number

  Recommended value is ``physical_cores / j``.  You can also tune it to
  achieve better performance.
* ``-j``, ``--intra_parallel``: [number] Thread Number Per-Instance  

  Avoid binding cores across numa nodes. Recommended value is 4.

In ``user.conf``:

* ``bert.Offline.target_qps``: Adjust according to the performance
* ``bert.Server.target_qps``: Adjust according to the performance

  Check the ``mlperf_log_summary.txt`` and make adjustments based on reference
  test results, scaling it by
  ``ref_QPS * (core_number/core_number_of_ref)``. You can increase these
  ``target_qps`` values if the performance is better.

  For server mode, to ensure the result is valid, increase the ``target_qps``
  as much as possible to make ``99.00 percentile latency`` close to ``target_latency``.
  If the ``99.00 percentile latency`` exceeds the ``target_latency``, the results
  will be invalid.


### Get the Results

Check the performance log file ``./test_log/mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``


Check the accuracy log file ``./test_log/accuracy.txt``.

* Check the field ``f1``


Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

## Get started with DLRM
If you haven't already done so, pull the Intel optimized Docker image for DLRM using:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm
```

### Prerequisites
Use these commands to prepare the Deep Learning Recommendation Model (DLRM)
dataset and model on your host system:

```
mkdir dlrm
cd dlrm

# dataset contain:
#     day_fea_count.npz
#     terabyte_processed_test.bin
#
# Learn how to get the dataset from:
#      https://github.com/facebookresearch/dlrm
# You can also copy it using:
#      scp -r mlperf@10.112.230.156:/home/mlperf/dlrm_data/* dlrm/
#
# get the model using:
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### Set Up Environment
Follow these steps to set up the docker instance.

#### Start a Container
Use ``docker run`` to start a container with the optimized Docker image we pulled earlier.
Replace ``/path/of/dlrm`` with the ``dlrm`` folder path created earlier:

```
# for DLRM, please run it with HT off
docker run --name dlrm_2-1 --privileged -itd --net=host --ipc=host \
  -v /path/of/dlrm:/data/mlperf_data/dlrm intel/intel-optimized-pytorch:mlperf-inference-2.1-dlrm
```

#### Login to Docker Container
Login into a bashrc shell in the Docker instance.

```
docker exec -it dlrm_2-1 bash
```

### Run the Benchmark

```
cd code/dlrm-99.9/pytorch-cpu/

# mode: offline or server
# type: perf (performance) or acc (accuracy)

bash run_mlperf.sh --mode=offline --type=perf --dtype=int8

```

#### Tune Parameters

In ``setup_env_offline.sh`` (offline) or ``setup_env_server.sh`` (server) scripts,
you can tune these parameters:

* ``CPUS_PER_SOCKET``: adjust it according to your system
* ``CPUS_PER_PROCESS``: how many processes will be used:

  ``process-per-socket = CPUS_PER_SOCKET / CPUS_PER_PROCESS``

  Recommended values are ``CPUS_PER_SOCKET = CPUS_PER_PROCESS``

* ``CPUS_PER_INSTANCE``: recommended value is ``1``

  ```
  instance-per-process number = CPUS_PER_PROCESS / CPUS_PER_INSTANCE
  total-instance = instance-per-process * process-per-socket
  ```
* ``BATCH_SIZE``: you can try to tune it to get better performance.


In ``user.conf``:

* ``dlrm.Offline.target_qps``: Adjust according to the performance
* ``dlrm.Server.target_qps``: Adjust according to the performance

  Check the ``mlperf_log_summary.txt`` and make adjustments based on reference
  test results, scaling it by
  ``ref_QPS * (core_number/core_number_of_ref)``. You can increase these
  ``target_qps`` values if the performance is better.

  For server mode, to ensure the result is valid, increase the ``target_qps``
  as much as possible to make ``99.00 percentile latency`` close to ``target_latency``.
  If the ``99.00 percentile latency`` exceeds the ``target_latency``, the results
  will be invalid.

### Get the Results

Check the appropriate offline or server performance log file, either
``./output/pytorch-cpu/dlrm/offline/performance/run_1/mlperf_log_summary.txt`` or
``./output/pytorch-cpu/dlrm/server/performance/run_1/mlperf_log_summary.txt``:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``

Check the appropriate offline or server accuracy log file, either
``./output/pytorch-cpu/dlrm/offline/accuracy/accuracy.txt`` or
``./output/pytorch-cpu/dlrm/server/accuracy/accuracy.txt``:

* Check the field ``roc_auc``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

##  Get Started with ResNet50
If you haven't already done so, pull the Intel optimized Docker image for ResNet50 using:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50
```

### Prerequisites
Use these commands to prepare the ResNet50 dataset and model on your host system:

```
# ImageNet(50000) validation datatset
bash download_imagenet.sh

```
### Set Up Environment
Follow these steps to set up the docker instance and preprocess data.

#### Start a Container

```
docker run --name resnet50_2-1 -v /path/to/ILSVRC2012_img_val:/opt/workdir/code/resnet50/pytorch-cpu/ILSVRC2012_img_val --privileged -itd --net=host \
  --ipc=host intel/intel-optimized-pytorch:mlperf-inference-2.1-resnet50
```

#### Login to Docker Instance
Login into a bashrc shell in the Docker instance.

```
docker exec -it resnet50_2-1 bash
cd code/resnet50/pytorch-cpu
```

If you need a proxy to access the internet, replace ``your host proxy`` with
the proxy server for your environment. If no proxy is needed, you can skip this step:
```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```
#### Prepare Calibration Dataset & Download Model
```
#prepare calibration 500 images into folders
bash prepare_calibration_dataset.sh

#model
bash download_model.sh
```

#### Quantize Torchscript Model and Check Accuracy

```
# set path
export DATA_CAL_DIR=calibration_dataset
export CHECKPOINT=resnet50-fp32-model.pth

# Generate scales and models
bash generate_torch_model.sh

# The start and end parts of the model are also saved (respectively named) in models
```

### Run the Benchmark

```
export DATA_DIR=${PWD}/ILSVRC2012_img_val
export RN50_START=models/resnet50-start-int8-model.pth
export RN50_END=models/resnet50-end-int8-model.pth
export RN50_FULL=models/resnet50-full.pth

# Run one of these performance or accuracy scripts at a time
# since the log files will be overwritten on each run

# for offline performance
bash run_offline.sh

# or for server performance
bash run_server.sh

# for offline accuracy
bash run_offline_accuracy.sh
python -u ./mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
  --mlperf-accuracy-file mlperf_log_accuracy.json \
  --imagenet-val-file ${DATA_DIR}/val_map.txt \
  --dtype int32 2>&1 | tee accuracy.txt

# or for server accuracy
bash run_server_accuracy.sh
python -u ./mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
  --mlperf-accuracy-file mlperf_log_accuracy.json \
  --imagenet-val-file ${DATA_DIR}/val_map.txt \
  --dtype int32 2>&1 | tee accuracy.txt
```

#### Tune Parameters

In ``run_offline.sh``, ``run_server.sh``, ``run_offline_accuracy.sh``,
and ``run_server_accuracy.sh`` scripts, you can tune these parameters:

* ``--num_instance 224``

  set ``num_instances`` according to your hardware:
  * for offline, according to number of Logical cores,
  * for server, set to ``physical_cores / cpus_per_instance``

* ``--warmup_iters 20``
* ``--cpus_per_instance 1``

  * for offline, recommend set to 1,
  * for server, recommend set to 4

* ``--total_sample_count 50000``
* ``--batch_size 9``


### Get the Results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``

Check the ``./accuracy.txt`` log file:

* Check the field ``accuracy``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

---

##  Get Started with Retinanet
If you haven't already done so, pull the Intel optimized Docker image for Retinanet using:
```
docker pull intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet
```

### Prerequisites
Use these commands to prepare the Retinanet dataset and model on your host system:

```
# Download openimages(264) dataset
export WORKLOAD_DATA=${PWD}/data
mkdir -p $WORKLOAD_DATA
bash openimages_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages

# Download Calibration images
bash openimages_calibration_mlperf.sh --dataset-path ${WORKLOAD_DATA}/openimages-calibration

# Download model
wget --no-check-certificate \
  'https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth' \
  -O 'retinanet-model.pth'
mv 'retinanet-model.pth' ${WORKLOAD_DATA}/
```
### Set Up Environment
Follow these steps to set up the Docker instance and preprocess data.

#### Start a Container

```
docker run --name retinanet_2-1 --privileged -itd --net=host \
  --ipc=host intel/intel-optimized-pytorch:mlperf-inference-2.1-datacenter-retinanet
```

#### Login into Docker Instance
Login into a bashrc shell in the docker instance.

```
docker exec -it retinanet_2-1 bash
cd code/retinanet/pytorch-cpu'
```

If you need a proxy to access the internet, replace ``your host proxy``
with the proxy server for your environment. If no proxy is needed,
you can skip this step:

```
export http_proxy="your host proxy"
export https_proxy="your host proxy"
```


#### Calibrate and Generate TorchScript Model

```
export CALIBRATION_DATA_DIR=${WORKLOAD_DATA}/openimages-calibration/train/data
export MODEL_CHECKPOINT=${WORKLOAD_DATA}/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=${WORKLOAD_DATA}/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
```

### Run the Benchmark

```
export DATA_DIR=${WORKLOAD_DATA}/openimages
export MODEL_PATH=${WORKLOAD_DATA}/retinanet-int8-model.pth

# Run one of these performance or accuracy scripts at a time
# since the log files will be overwritten on each run

# for offline performance
bash run_offline.sh

# for server performance
bash run_server.sh

# for offline accuracy
bash run_offline_accuracy.sh

# for server accuracy
bash run_server_accuracy.sh
```

#### Tune Parameters

In ``run_offline.sh``, ``run_server.sh``, ``run_offline_accuracy.sh``,
and ``run_server_accuracy.sh`` scripts, you can tune these parameters:

* ``--cpus_per_instance``

  * For offline, recommended value is 4.
  * For server, recommended value is 8.

* ``--num_instance``

  set num_instances according to your hardware:
  * set to ``physical_cores / cpus_per_instance``

* ``--batch_size 2``

### Get the results

Check the ``./mlperf_log_summary.txt`` log file:

* Verify you see ``results is: valid``.
* For offline mode performance, check the field ``Samples per second:``
* For server mode performance, check the field ``Scheduled samples per second:``

Check the ``./accuracy.txt`` log file:

* Check the field ``mAP``

Save these output log files elsewhere when each test is completed as they will be overwritten by the next test.

## Results on 2S SPR 56C for v2.1


|	Model	   |	Scenario	|	2S SPR (v2.1)		|
|	-----------	|	-----------	|	-----------		|
|	ResNet50	|	Offline	|	16179		|
|		|	Server	|	11846		|
|	Retinanet	|	Offline	|	232		|
|		|	Server	|	150		|
|	3DUNet	|	Offline	|	1.47054		|
|	BERT	|	Offline	|	1373.47		|
|		|	Server	|	1097.01		|
|	DLRM	|	Offline	|	106444		|
|		|	Server	|	73994.6		|
|	RNNT	|	Offline	|	3401		|
|		|	Server	|	1207.32		|


