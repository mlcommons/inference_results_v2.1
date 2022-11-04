#!/bin/bash

CONDA_ENV_NAME=/opt/conda
source /opt/conda/etc/profile.d/conda.sh
conda activate ${CONDA_ENV_NAME}


if [ -z "${CHECKPOINT}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export CHECKPOINT="
    exit 1
fi

if [ -z "${DATA_CAL_DIR}" ]; then
    echo "Path to annotations for calibration images not set. Please set it:"
    echo "export DATA_CAL_DIR="
    exit 1
fi
CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# echo "workspace is ${CUR_DIR}"
export ARGS="--batch-size 1 --data-path-cal ${DATA_CAL_DIR} --checkpoint-path ${CHECKPOINT} --save-dir ${CUR_DIR}/models --calibrate-start-partition --calibrate-end-partition --calibrate-full-weights --save-full-weights --channels-last --massage"

numactl -m 0 python -u main.py ${ARGS}
