#!/usr/bin/env bash

git submodule update --init --recursive

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=/opt/conda
export WORKDIR=${PWD}

echo "Working directory is ${WORKDIR}"
source /opt/conda/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
echo "CMAKE_PREFIX_PATH is ${CMAKE_PREFIX_PATH}"
export IPEX_PATH=/opt/workdir/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch
export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
export LOADGEN_DIR=${WORKDIR}/mlperf_inference/loadgen
export OPENCV_DIR=${WORKDIR}/opencv/build
export RAPIDJSON_INCLUDE_DIR=${WORKDIR}/rapidjson/include
export GFLAGS_DIR=${WORKDIR}/gflags/build
export ONEDNN_DIR=${WORKDIR}/oneDNN

# cd ${CUR_DIR}
# ============================
echo "=== Building binaries ==="
BUILD_DIR=${WORKDIR}/build
SRC_DIR=${WORKDIR}/src
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${OPENCV_DIR}/lib:${ONEDNN_DIR}/build/src:${CONDA_PREFIX}/lib
echo "LD_LIBRARY_PATH is ${LD_LIBRARY_PATH}"
export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib

cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
    -DLOADGEN_DIR=${LOADGEN_DIR} \
    -DOpenCV_DIR=${OPENCV_DIR} \
    -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
    -Dgflags_DIR=${GFLAGS_DIR} \
    -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
    -DONEDNN_DIR=${ONEDNN_DIR} \
    -B${BUILD_DIR} \
    -H${SRC_DIR}

echo "step 1 finished"

cmake --build ${BUILD_DIR} --config Release -j$(nproc)

echo "step 2 finished"
