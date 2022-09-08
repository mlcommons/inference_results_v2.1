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
echo -e "\n\n***************************************************************"
echo -e "***************************************************************"
echo "CURRENT_PATH:$CURRENT_PATH , basename:$0 , system_name:$1"

# every path is relative path from $BUILD_DIR
BUILD_DIR=./build
cd $BUILD_DIR

ROOT_DIR=../../../../../..
SUBMISSION_ROOT=../../..
MLPERF_DIR=../MlperfInference
MEASUREMENTS_DIR=$SUBMISSION_ROOT/measurements
RESULTS_DIR=$SUBMISSION_ROOT/results

data_dir=../data
input_data_dir=$data_dir/preprocessed_data
annotation_path=$data_dir/val.txt
weight_path=$data_dir/weight_result

if [ ! -e $input_data_dir ];then
    echo "[ERR] $input_data_dir not exists. Check code/resnet50/preprocess_data.sh and uncomment to redownload required data"
    exit 1
fi
if [ ! -e $annotation_path ];then
    echo "[ERR] $annotation_path not exists. Check code/resnet50/preprocess_data.sh and uncomment to redownload required data"
    exit 1
fi
if [ ! -e $weight_path ];then
    echo "[ERR] $weight_path not exists. Check code/utils/README.md and run code/utils/run.sh script"
    exit 1
fi

system_name=$1
model=resnet50

if [ "$system_name" == "X220-compact" ]
then
    echo "this is X220-compact device"
    run_exe="taskset --cpu-list 0-10 ./sapeon_inferencer --device $system_name"
else
    echo "this is X220-enterprise device"
    run_exe="taskset --cpu-list 0-15 ./sapeon_inferencer --device $system_name"
fi

accuracy_checker=$MLPERF_DIR/vision/classification_and_detection/tools/accuracy-imagenet.py

# reset results dir
rm -rf $RESULTS_DIR
mkdir $RESULTS_DIR

# remove unexpected file `audit.config`
AUDIT_FILENAME=audit.config
if [ -f "$AUDIT_FILENAME" ] ; then
    echo "$AUDIT_FILENAME exists! remove it.."
    rm $AUDIT_FILENAME
fi

for scenario in Offline Server
do
    mkdir -p $RESULTS_DIR/$system_name/$model/$scenario/accuracy
    echo "   $run_exe -s $scenario -m AccuracyOnly -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf \
        -o $RESULTS_DIR/$system_name/$model/$scenario/accuracy"
    $run_exe -s $scenario -m AccuracyOnly -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf \
        -o $RESULTS_DIR/$system_name/$model/$scenario/accuracy

    python3 $accuracy_checker --mlperf-accuracy-file $RESULTS_DIR/$system_name/$model/$scenario/accuracy/mlperf_log_accuracy.json \
        --imagenet-val-file $annotation_path --dtype int32 \
        > $RESULTS_DIR/$system_name/$model/$scenario/accuracy/accuracy.txt

    mkdir -p $RESULTS_DIR/$system_name/$model/$scenario/performance/run_1
    echo "   $run_exe -s $scenario -m PerformanceOnly -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf \
        -o $RESULTS_DIR/$system_name/$model/$scenario/performance/run_1"
    $run_exe -s $scenario -m PerformanceOnly -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf \
        -o $RESULTS_DIR/$system_name/$model/$scenario/performance/run_1
done

find $RESULTS_DIR -name mlperf_log_trace.json -delete
find $RESULTS_DIR/$system_name/$model/Offline/performance/ -name mlperf_log_accuracy.json -delete
find $RESULTS_DIR/$system_name/$model/Server/performance/ -name mlperf_log_accuracy.json -delete


echo "$0 complete"
