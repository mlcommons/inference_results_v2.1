#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 [X220-compact|X220-enterprise]"
    exit 1
fi

if [ "$1" != "X220-compact" ] && [ "$1" != "X220-enterprise" ]
then
    echo "argument : [$1] must be [X220-compact] or [X220-enterprise]"
    exit 1
fi

CURRENT_PATH=$(pwd)
echo "CURRENT_PATH:$CURRENT_PATH , basename:$0 , system_name:$1"

system_name=$1

MLPERF_DIR="MlperfInference"
if [ -d "$MLPERF_DIR" ] ; then
    echo "$MLPERF_DIR already exists! auto remove start.."
    rm -rf $MLPERF_DIR
    echo "remove complete"
fi
git clone -b r2.1 --single-branch https://github.com/mlcommons/inference.git MlperfInference 

if [ -e "build" ]; then
    rm -rf build
fi
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j`nproc`
cd ..
chmod -R 777 build

./preprocess_data.sh
./run_system.sh $system_name

# give all permission to docker results
chmod -R 777 /submission