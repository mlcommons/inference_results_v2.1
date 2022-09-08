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
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

function run_cmd()
{
  local cmdline=$*
  echo "run command: "${cmdline}
  ${cmdline}
}

if [ "$#" -lt 4 ]; then
  echo "Use command: bert_99.9_compliance.sh [perf|accu|sub] [Offline|Server] [systemid] [TEST01|TEST05]"
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
testid="$4"

deviceslist="0"
if [[ $systemid =~ "x8" ]]
then
  echo "has 8 devices"
  deviceslist="0,1,2,3,4,5,6,7"
fi

if [[ $systemid =~ "x1" ]]
then
  deviceslist="0"
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

logdir="/work/compliance/"${systemid}"/bert-99.9/"${scenario}"/"${testid}"/"${testLog}
mkdir -p ${logdir}

unlink vectors
ln -s /work/build/data/bert_large_99.9/vectors/ vectors

# walk around to disable suinfer general json for multiple instance usage
export SUIN_DISABLE_GENERAL_JSON=true

# work around of umd
export DISABLE_L2PAD=1

run_cmd "rm -f audit.config"

if [ ${testid}x = "TEST01"x ]; then
  run_cmd "cp build/inference/compliance/nvidia/${testid}/bert/audit.config ."
else
  run_cmd "cp build/inference/compliance/nvidia/${testid}/audit.config ."
fi

make build_harness  &&  /work/build/bin/harness_bert \
    --logfile_outdir=${logdir} \
    --logfile_prefix="mlperf_log_" \
    --performance_sample_count=10833 \
    --gpu_batch_size=10833 \
    --devices=${deviceslist} \
    --verbose=true \
    --max_packing_size=3 \
    --map_path="data_maps/squad/val_map.txt" \
    --tensor_path="/work/build/preprocessed_data/bert_data_int32/input_ids/,/work/build/preprocessed_data/bert_data_int32/segment_ids/,/work/build/preprocessed_data/bert_data_int32/input_mask/" \
    --mlperf_conf_path="measurements/"${systemid}"/bert-99.9/"${scenario}"/mlperf.conf" \
    --user_conf_path="measurements/"${systemid}"/bert-99.9/"${scenario}"/user.conf" \
    --scenario ${scenario} \
    --test_mode=${testmode} \
    --model bert \
    --model_path="/work/build/models/bert/bert_large_99.9_4inputs.onnx"

if [ ${testmode} = "AccuracyOnly" -o ${testmode} = "SubmissionRun" ]; then
  export PYTHONPATH=/work/build/inference/language/bert/DeepLearningExamples/TensorFlow/LanguageModeling/BERT:$PYTHONPATH
  cmd_line="python3 build/inference/language/bert/accuracy-squad.py \
            --log_file ${logdir}/mlperf_log_accuracy.json \
            --vocab_file data_maps/squad/vocab.txt \
            --val_data data_maps/squad/dev-v1.1.json \
            --out_file ${logdir}/predictions.json \
            --output_dtype float32"
  run_cmd ${cmd_line}
  run_cmd "rm -f ${logdir}/predictions.json"
fi
