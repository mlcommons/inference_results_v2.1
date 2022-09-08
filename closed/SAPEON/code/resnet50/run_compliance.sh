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
COMPLIANCE_DIR=$SUBMISSION_ROOT/compliance
RESULTS_DIR=$SUBMISSION_ROOT/results
MEASUREMENTS_DIR=$SUBMISSION_ROOT/measurements

data_dir=../data
input_data_dir=$data_dir/preprocessed_data
annotation_path=$data_dir/val.txt
weight_path=$data_dir/weight_result

test01_dir=$MLPERF_DIR/compliance/nvidia/TEST01
test04_dir=$MLPERF_DIR/compliance/nvidia/TEST04
test05_dir=$MLPERF_DIR/compliance/nvidia/TEST05

system_name=$1
model=resnet50

COMPLIANCE_LOG_PATH=$ROOT_DIR/compliance_log

if [ "$system_name" == "X220-compact" ]
then
    echo "this is X220-compact device"
    run_exe="taskset --cpu-list 0-10 ./sapeon_inferencer --device $system_name"
else
    echo "this is X220-enterprise device"
    run_exe="taskset --cpu-list 0-15 ./sapeon_inferencer --device $system_name"
fi

# reset compilance dir
rm -rf $COMPLIANCE_DIR 
rm -rf $COMPLIANCE_LOG_PATH
mkdir $COMPLIANCE_DIR

# TEST01
for scenario in Offline Server
do
    echo -e "\n\n***************************************************************"
    echo -e "***************************** compliance test TEST01 [$scenario]\n"
    log_dir=$COMPLIANCE_LOG_PATH/$scenario/TEST01_log
    mkdir -p $log_dir
    mlperf_compilance_test_dir=$test01_dir
    output_dir=$COMPLIANCE_DIR/$system_name/$model/$scenario
    cp $mlperf_compilance_test_dir/resnet50/audit.config audit.config
    echo "$run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf"
    $run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf 
    echo -e "\n"
    python3 $mlperf_compilance_test_dir/run_verification.py -r $RESULTS_DIR/$system_name/$model/$scenario -c $log_dir -o $output_dir --dtype int32
    rm audit.config
done

# TEST04
for scenario in Offline Server
do
    echo -e "\n\n***************************************************************"
    echo -e "***************************** compliance test TEST04 [$scenario]\n"
    log_dir=$COMPLIANCE_LOG_PATH/$scenario/TEST04_log
    mkdir -p $log_dir
    mlperf_compilance_test_dir=$test04_dir
    output_dir=$COMPLIANCE_DIR/$system_name/$model/$scenario
    cp $mlperf_compilance_test_dir/audit.config audit.config
    echo "    $run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf "
    $run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf 
    echo -e "\n"
    python3 $mlperf_compilance_test_dir/run_verification.py -r $RESULTS_DIR/$system_name/$model/$scenario -c $log_dir -o $output_dir
    rm audit.config
done

# TEST05
for scenario in Offline Server
do
    echo -e "\n\n***************************************************************"
    echo -e "***************************** compliance test TEST05 [$scenario]\n"
    log_dir=$COMPLIANCE_LOG_PATH/$scenario/TEST05_log
    mkdir -p $log_dir
    mlperf_compilance_test_dir=$test05_dir
    output_dir=$COMPLIANCE_DIR/$system_name/$model/$scenario
    cp $mlperf_compilance_test_dir/audit.config audit.config
    echo "    $run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf "
    $run_exe -o $log_dir -s $scenario -i $input_data_dir -l $annotation_path -b $weight_path \
        -c $MEASUREMENTS_DIR/$system_name/$model/$scenario/user.conf \
        -mc $MEASUREMENTS_DIR/$system_name/$model/$scenario/mlperf.conf 
    echo -e "\n"
    python3 $mlperf_compilance_test_dir/run_verification.py -r $RESULTS_DIR/$system_name/$model/$scenario -c $log_dir -o $output_dir
    rm audit.config
done

echo "$0 complete"
