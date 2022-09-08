#!/bin/bash

if [[ -z $HARDWARE_PLATFORM ]]; then
    echo "PLEASE SET HARDWARE_PLATFORM !!! "
    exit 1
fi

# TEST01
cp /home/moffett/mlcommons/inference/compliance/nvidia/TEST01/bert/audit.config ./ && \
python3 run_bert.py --config ../../../config/bert_offline_${HARDWARE_PLATFORM,,}.yaml --mode AccuracyOnly --output_dir TEST01 && \
python3 run_bert.py --config ../../../config/bert_offline_${HARDWARE_PLATFORM,,}.yaml --mode PerformanceOnly --output_dir TEST01 && \
rm ./audit.config && \
python3 /home/moffett/mlcommons/inference/compliance/nvidia/TEST01/run_verification.py -r ./ -c TEST01 -o ../../../../results/SPARSEONE_${HARDWARE_PLATFORM^^}/bert-99.9/Offline --dtype int32 && \

# TEST05
cp /home/moffett/mlcommons/inference/compliance/nvidia/TEST05/audit.config ./ && \
python3 run_bert.py --config ../../../config/bert_offline_${HARDWARE_PLATFORM,,}.yaml --mode AccuracyOnly --output_dir TEST05 && \
python3 run_bert.py --config ../../../config/bert_offline_${HARDWARE_PLATFORM,,}.yaml --mode PerformanceOnly --output_dir TEST05 && \
rm ./audit.config && \
python3 /home/moffett/mlcommons/inference/compliance/nvidia/TEST05/run_verification.py -r ./ -c TEST05 -o ../../../../results/SPARSEONE_${HARDWARE_PLATFORM^^}/bert-99.9/Offline