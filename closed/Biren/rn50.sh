#
# Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

function run_cmd()
{
  local cmdline=$*
  echo "run command: "${cmdline}
  ${cmdline}
}

if [ "$#" -lt 3 ]; then
  echo "Use command: rn50.sh [perf|accu|sub] [Offline|Server] [systemid]"
  exit 1
fi

testmode="PerformanceOnly"
testLog="performance/run_1"

if [ "$1" = "perf" ]; then
  testmode="PerformanceOnly"
  testLog="performance/run_1"
fi

if [ "$1" = "accu" ]; then
  testmode="AccuracyOnly"
  testLog="accuracy"
fi

if [ "$1" = "sub" ]; then
  testmode="SubmissionRun"
fi

scenario="$2"
systemid="$3"

deviceslist="0"
if [[ $systemid =~ "x8" ]]
then
  echo "has 8 devices"
  deviceslist="0,1,2,3,4,5,6,7"
fi

if [[ $systemid =~ "x1" ]]
then
  deviceslist="1"
fi

if [[ $systemid =~ "x2" ]]
then
  deviceslist="0,1"
fi

if [[ $systemid =~ "x4" ]]
then
  deviceslist="0,1,2,3"
fi

echo "Running in "${testmode}" mode on devices "${deviceslist}

logdir="/work/results/"${systemid}"/resnet50/"${scenario}"/"${testid}"/"${testLog}
mkdir -p ${logdir}

unlink vectors
if [ "$scenario" = "Server" ]; then
  # for small batch parallel run kernel
  # Set following environment variable to make every CP queue submission wait for read ptr update before return.
  echo "running in server mode"
  export ENABLE_READ_PTR_WAIT=1
  ln -s /work/build/data/resnet50_small_batch vectors
else
  ln -s /work/build/data/resnet50 vectors
fi

# walk around to disable suinfer general json for multiple instance usage
export SUIN_DISABLE_GENERAL_JSON=true

run_cmd "rm -f audit.config"

make build_harness && /work/build/bin/harness_default \
    --logfile_outdir=${logdir} \
    --logfile_prefix="mlperf_log_" \
    --performance_sample_count=2048 \
    --gpu_batch_size=2048 \
    --devices=${deviceslist} \
    --verbose=false \
    --map_path="data_maps/imagenet/val_map.txt" \
    --tensor_path="/work/build/preprocessed_data/imagenet_int8_new_block0/" \
    --mlperf_conf_path="measurements/"${systemid}"/resnet50/${scenario}/mlperf.conf" \
    --user_conf_path="measurements/"${systemid}"/resnet50/${scenario}/user.conf" \
    --scenario ${scenario} \
    --test_mode=${testmode} \
    --model resnet50 \
    --model_path="/work/build/models/resnet50/resnet50_int8_delete_quantize_linear_v2.onnx"

if [ ${testmode} = "AccuracyOnly" -o ${testmode} = "SubmissionRun" ]; then
  accuracycmd_1="python3 build/inference/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file "
  accuracycmd_2=" --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32"

  cmd_line="${accuracycmd_1} ${logdir}/mlperf_log_accuracy.json ${accuracycmd_2}"
  ${cmd_line} > ${logdir}/accuracy.txt
  cat ${logdir}/accuracy.txt
fi
