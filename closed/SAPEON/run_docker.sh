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

echo -e "\n\n***************************************************************"
echo -e "*** run mlperf with docker"
echo ""
echo ""

# get docker image. if you want download docker image uncomment below lines
# wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/sapeon_runtime_docker_img.tar
# docker load -i sapeon_runtime_docker_img.tar
# rm sapeon_runtime_docker_img.tar

# get Sapeon Runtime .so file
wget https://cloud.sapeon.net:5043/mlperf/resnet50/recent_data/lib/libsapeonruntime.so.0
mkdir code/resnet50/lib
mv libsapeonruntime.so.0 code/resnet50/lib/


if [ $system_name == "X220-compact" ]
then
    # run mlperf inference and compliance test
    cd ../../
    docker run -it --rm -v $PWD/:/submission \
        -v /home/shared:/home/shared \
        --device /dev/aix0 -w /submission/closed/SAPEON/code/resnet50/ \
        sapeoninc/ubuntu-dev:1.0 ./run.sh $system_name

    # run "truncate_accuracy_log.py" and "submission-checker.py"
    cd closed/SAPEON/code/resnet50
    ./run_complete.sh ../../../../../SAPEON_SUBMISSION_X220-compact_checked

    echo "[SAPEON_SUBMISSION_X220-compact_checked] created at outside of submission dir. If not check log.log"
fi
if [ $system_name == "X220-enterprise" ]
then
    # run mlperf inference and compliance test
    cd ../../
    docker run -it --rm -v $PWD/:/submission \
        -v /home/shared:/home/shared \
        --device /dev/aix0 --device /dev/aix1 -w /submission/closed/SAPEON/code/resnet50/ \
        sapeoninc/ubuntu-dev:1.0 ./run.sh $system_name

    # run "truncate_accuracy_log.py" and "submission-checker.py"
    cd closed/SAPEON/code/resnet50
    ./run_complete.sh ../../../../../SAPEON_SUBMISSION_X220-enterprise_checked

    echo "[SAPEON_SUBMISSION_X220-enterprise_checked] created at outside of submission dir. If not check log.log file"
fi
