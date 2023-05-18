#!/bin/bash
export CALIBRATION_DATA_DIR=/workspace/data/openimages-calibration/train/data
export MODEL_CHECKPOINT=/workspace/data/retinanet-model.pth
export CALIBRATION_ANNOTATIONS=/workspace/data/openimages-calibration/annotations/openimages-mlperf-calibration.json
bash run_calibration.sh
echo "step 1 done!"
export DATA_DIR=/workspace/data/openimages
export MODEL_PATH=/workspace/data/retinanet-int8-model.pth

number_threads=`nproc --all`
number_cores=$((number_threads/2))
number_instance=$((number_threads/4))
number_sockets=`grep physical.id /proc/cpuinfo | sort -u | wc -l`
cpu_per_socket=$((number_cores/number_sockets))

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_PATH=</path/to/retinanet-int8-model.pth>"
    exit 1
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

#export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP=${PWD}/build/bin/mlperf_runner

if [ -e "mlperf_log_summary.txt" ]; then
    rm mlperf_log_summary.txt
fi

${APP} --scenario Server \
	--mode Performance \
	--mlperf_conf mlperf.conf \
	--user_conf user.conf \
	--model_name retinanet \
    --model_path ${MODEL_PATH} \
	--data_path ${DATA_DIR} \
	--num_instance $number_instance \
	--warmup_iters 100 \
	--cpus_per_instance 4 \
	--total_sample_count 24781 \
    --batch_size 2
	
if [ -e "mlperf_log_summary.txt" ]; then
    cat mlperf_log_summary.txt
fi
