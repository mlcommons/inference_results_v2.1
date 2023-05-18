#!/bin/bash

number_threads=`nproc --all`
number_cores=$((number_threads/2))
number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
cpu_per_socket=$((number_cores/number_sockets))

export DATA_CAL_DIR=/workspace/calibration_dataset
export CHECKPOINT=/workspace/resnet50-fp32-model.pth

bash /workspace/generate_torch_model.sh
bash /workspace/build_binaries.sh
echo "step 3 finished"

export DATA_DIR=/workspace/ILSVRC2012_img_val
export RN50_START=/workspace/models/resnet50-start-int8-model.pth
export RN50_END=/workspace/models/resnet50-end-int8-model.pth
export RN50_FULL=/workspace/models/resnet50-full.pth


if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${RN50_START}" ]; then
    echo "Path to resnet50_start model not set. Please set it:"
    export "RN50_START=</path/to/resnet50_start.pth>"
    exit 1
fi

if [ -z "${RN50_END}" ]; then
    echo "Path to resnet50_end model not set. Please set it:"
    export "RN50_END=</path/to/resnet50_end.pth>"
    exit 1
fi

if [ -z "${RN50_FULL}" ]; then
    echo "Path to resnet50_full model not set. Please set it:"
    export "RN50_FULL=</path/to/resnet50_full.pth>"
    exit 1
fi

# comment out the follwoing lines for AWS
# CONDA_ENV_NAME=rn50-mlperf
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate ${CONDA_ENV_NAME}

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP=/opt/workdir/code/resnet50/pytorch-cpu/build/bin/mlperf_runner

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

if [ -e mlperf_log_accuracy.json ]; then
    rm mlperf_log_accuracy.json
fi

numactl -m 0 ${APP} --scenario Server \
    --mode Accuracy \
    --mlperf_conf ${CUR_DIR}/src/mlperf.conf \
    --user_conf ${CUR_DIR}/src/user.conf \
    --model_name resnet50 \
    --rn50-part1 ${RN50_START} \
    --rn50-part3 ${RN50_END} \
    --rn50-full-model ${RN50_FULL} \
    --data_path ${DATA_DIR} \
    --num_instance $number_threads \
    --warmup_iters 50 \
    --cpus_per_instance 4 \
    --total_sample_count 50000 \
    --batch_size 4

echo " ==================================="
echo "         Evaluating Accuracy        "
echo " ==================================="

if [ -e mlperf_log_accuracy.json ]; then
    python -u ${CONDA_ENV_NAME}/mlperf_inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
        --mlperf-accuracy-file mlperf_log_accuracy.json \
        --imagenet-val-file ${DATA_DIR}/val_map.txt \
        --dtype int32
fi

