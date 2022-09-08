#!/bin/bash

if [[ -z $HARDWARE_PLATFORM ]]; then
    echo "PLEASE SET HARDWARE_PLATFORM !!! "
    exit 1
fi

# TEST01
cp /home/moffett/mlcommons/inference/compliance/nvidia/TEST01/resnet50/audit.config ./ && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode AccuracyOnly --output_dir TEST01 && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode PerformanceOnly --output_dir TEST01 && \
rm ./audit.config && \
python3 /home/moffett/mlcommons/inference/compliance/nvidia/TEST01/run_verification.py -r ./ -c TEST01 -o ../../../../results/SPARSEONE_${HARDWARE_PLATFORM^^}/resnet50/Offline --dtype int32 && \

# TEST04
cp /home/moffett/mlcommons/inference/compliance/nvidia/TEST04/audit.config ./ && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode AccuracyOnly --output_dir TEST04 && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode PerformanceOnly --output_dir TEST04 && \
rm ./audit.config && \
python3 /home/moffett/mlcommons/inference/compliance/nvidia/TEST04/run_verification.py -r ./ -c TEST04 -o ../../../../results/SPARSEONE_${HARDWARE_PLATFORM^^}/resnet50/Offline && \

# TEST05
cp /home/moffett/mlcommons/inference/compliance/nvidia/TEST05/audit.config ./ && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode AccuracyOnly --output_dir TEST05 && \
python3 run_resnet50.py --config ../../../config/resnet50_offline_${HARDWARE_PLATFORM,,}.yaml --mode PerformanceOnly --output_dir TEST05 && \
rm ./audit.config && \
python3 /home/moffett/mlcommons/inference/compliance/nvidia/TEST05/run_verification.py -r ./ -c TEST05 -o ../../../../results/SPARSEONE_${HARDWARE_PLATFORM^^}/resnet50/Offline