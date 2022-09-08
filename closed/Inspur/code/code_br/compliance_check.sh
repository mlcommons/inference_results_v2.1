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

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0

function run_cmd()
{
  local cmdline=$*
  echo "run command: "${cmdline}
  ${cmdline}
  echo ==================================================================
}

function checkPerfCompliance()
{
  local resultDir=$1
  local complianceDir=$2

  cmdline="python3 build/inference/compliance/nvidia/TEST01/verify_performance.py \
          -r ${resultDir}/mlperf_log_summary.txt \
          -t ${complianceDir}/mlperf_log_summary.txt "
  result_file="${complianceDir}/../../verify_performance.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"
}

function checkBertAccuracy()
{
  local resultDir=$1
  local complianceDir=$2

  mkdir ${complianceDir}/../../accuracy
  echo ==================================

  # Step1: Comparing the accuracy in compliance result with accuracy in result
  cmdline="python3 build/inference/compliance/nvidia/TEST01/verify_accuracy.py \
          --reference_accuracy ${resultDir}/mlperf_log_accuracy.json \
          --test_accuracy  ${complianceDir}/mlperf_log_accuracy.json \
          --dtype=float32"
  result_file="${complianceDir}/../../verify_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"

  re=$(grep 'TEST ' ${result_file})
  echo ${re}
  if [[ ${re} =~ "TEST PASS" ]]; then
    echo "Accuracy test passed and return."
    return
  fi

  # Step2: generate accuracy baseline file
  cmdline="bash build/inference/compliance/nvidia/TEST01/create_accuracy_baseline.sh \
           ${resultDir}/mlperf_log_accuracy.json \
           ${complianceDir}/mlperf_log_accuracy.json"
  run_cmd ${cmdline}

  # Step3: get accuracy file for mlperf_log_accuracy.log in compliance folder and move
  #        checking log (baseline_accuracy.txt) to accuracy folder of compliance
  cmdline="python3 build/inference/language/bert/accuracy-squad.py \
           --log_file mlperf_log_accuracy_baseline.json \
           --vocab_file data_maps/squad/vocab.txt \
           --val_data data_maps/squad/dev-v1.1.json \
           --out_file ${complianceDir}/predictions.json \
           --output_dtype float32"
  result_file="${complianceDir}/../../accuracy/baseline_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"

  # remove predictions.json
  run_cmd "rm -f ${complianceDir}/predictions.json"
  # remove generated mlperf_log_accuracy_baseline.json
  run_cmd "rm -f mlperf_log_accuracy_baseline.json"

  # Step4: get accuracy file for mlperf_log_accuracy_baseline.log in compliance folder and
  #        move checking log (compliance_accuracy.txt) to accuracy folder of compliance
  cmdline="python3 build/inference/language/bert/accuracy-squad.py \
           --log_file ${complianceDir}/mlperf_log_accuracy.json \
           --vocab_file data_maps/squad/vocab.txt \
           --val_data data_maps/squad/dev-v1.1.json \
           --out_file ${complianceDir}/predictions.json \
           --output_dtype float32"
  result_file="${complianceDir}/../../accuracy/compliance_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"
  run_cmd "rm -f ${complianceDir}/predictions.json"

  run_cmd "mv ${complianceDir}/mlperf_log_accuracy.json ${complianceDir}/../../accuracy/."
}

function checkImageAccuracy()
{
  local resultDir=$1
  local complianceDir=$2

  mkdir ${complianceDir}/../../accuracy

  # Step1: Comparing the accuracy in compliance result with accuracy in result
  cmdline="python3 build/inference/compliance/nvidia/TEST01/verify_accuracy.py \
          --reference_accuracy ${resultDir}/mlperf_log_accuracy.json \
          --test_accuracy  ${complianceDir}/mlperf_log_accuracy.json \
          --dtype=int32"
  result_file="${complianceDir}/../../verify_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"
  run_cmd "rm -f ${complianceDir}/predictions.json"
  run_cmd ${cmdline}

  re=$(grep 'TEST ' ${result_file})
  echo ${re}
  if [[ ${re} =~ "TEST PASS" ]]; then
    echo "Accuracy test passed and return."
    return
  fi

  # Step2: generate accuracy baseline file
  cmdline="bash build/inference/compliance/nvidia/TEST01/create_accuracy_baseline.sh \
           ${resultDir}/mlperf_log_accuracy.json \
           ${complianceDir}/mlperf_log_accuracy.json"
  run_cmd ${cmdline}

  # Step3: get accuracy file for mlperf_log_baseline_accuracy.log in compliance folder and move
  #        checking log (baseline_accuracy.txt) to accuracy folder of compliance
  cmdline="python3 build/inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
          --mlperf-accuracy-file mlperf_log_accuracy_baseline.json \
          --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32"
  result_file="${complianceDir}/../../accuracy/baseline_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"

  # remove generated accuracy baseline json file
  run_cmd "rm -f mlperf_log_accuracy_baseline.json"

  # Step4: get accuracy file for mlperf_log_accuracy_baseline.log in compliance folder and
  #        move checking log (compliance_accuracy.txt) to accuracy folder of compliance
  cmdline="python3 build/inference/vision/classification_and_detection/tools/accuracy-imagenet.py \
          --mlperf-accuracy-file ${complianceDir}/mlperf_log_accuracy.json \
          --imagenet-val-file data_maps/imagenet/val_map.txt --dtype int32"
  result_file="${complianceDir}/../../accuracy/compliance_accuracy.txt"
  run_cmd ${cmdline} > ${result_file}
  run_cmd "cat ${result_file}"

  run_cmd "mv ${complianceDir}/mlperf_log_accuracy.json ${complianceDir}/../../accuracy/."
}

# Only TEST01 has accuracy compliance check
function checkAccuracyCompliance()
{
  # bert-99.9|bert-99|resnet50
  local model=$1
  # Offline|Server
  local testmod=$2
  # BR104-300W-PCIexN
  local systemid=$3
  # results
  local resultDir=$4
  # compliance
  local complianceDir=$5

  local compliancefolder="${complianceDir}/${systemid}/${model}/${testmod}/TEST01/performance/run_1/"
  local resultfolder="${resultDir}/${systemid}/${model}/${testmod}/accuracy/"
  
  if [ ${model}x = "resnet50"x ]; then
    echo "resnet50" "${resultfolder}" "${compliancefolder}"
    checkImageAccuracy "${resultfolder}" "${compliancefolder}"
  else
    echo "bert" "${resultfolder}" "${compliancefolder}"
    checkBertAccuracy "${resultfolder}" "${compliancefolder}"
  fi   
}


function checkPerformanceCompliance()
{
  # bert-99.9|bert-99|resnet50
  local model=$1
  # Offline|Server
  local testmod=$2
  # BR104-300W-PCIexN
  local systemid=$3
  # results
  local resultDir=$4
  # compliance
  local complianceDir=$5

  local subfolder="${systemid}/${model}/${testmod}/"
  local datafolder="performance/run_1/"
  testids=""
  if [ ${model}x = "resnet50"x ]; then
    testids="TEST01 TEST04 TEST05"
  else
    testids="TEST01 TEST05"
  fi

  for testid in ${testids}
  do
    echo "${resultDir}/${subfolder}/${datafolder}" "${complianceDir}/${subfolder}/${testid}/${datafolder}"
    checkPerfCompliance "${resultDir}/${subfolder}/${datafolder}" "${complianceDir}/${subfolder}/${testid}/${datafolder}"
  done
}

function checkCompliance()
{
  # bert-99.9|bert-99|resnet50
  local model=$1
  # BR104-300W-PCIexN
  local systemid=$2
  # results
  local resultDir=$3
  # compliance
  local complianceDir=$4

  for testmod in Offline Server
  do
    checkAccuracyCompliance ${model} ${testmod} ${systemid} ${resultDir} ${complianceDir}
    checkPerformanceCompliance ${model} ${testmod} ${systemid} ${resultDir} ${complianceDir}
  done
}

if [ "$#" -lt 3 ]; then
  echo "Use command: compliance_check.sh model path_to_results_folder path_to_compliance_folder systemids"
  echo "i.e.: compliance_check.sh model path_to_results_folder path_to_compliance_folder BR104-300W-PCIex8,BR104-300W-PCIex4"
  echo "model: model type to test, resnet50, bert99.9, ect."
  echo "path_to_results_folder: path to the result folder of normal running"
  echo "path_to_compliance_folder: path to the compliance folder with compliance audit config"
  # we shall enhance to get all subfolder from compliance folder
  echo "systemids: all system ids to perform compliance check, valid ids are BR104-300W-PCIex8 BR104-300W-PCIex4 BR104-300W-PCIex2 BR104-300W-PCIex1. By default. all will be checked."
  exit 1
fi

model=$1
resultDir=$2
complianceDir=$3
systemids="NF5468M6_BR104-300W-PCIex4"
echo "Num of systemids" $#
if [ "$#" -ge 4 ]; then
  systemids=$4
  systemids=${systemids//,/ }
fi

for systemid in ${systemids}
do
  checkCompliance ${model} ${systemid} ${resultDir} ${complianceDir}
done

cmdline="python3 ../mlperf-submission/build/inference/tools/submission/truncate_accuracy_log.py \
         --input . \
         --backup build/full_logs/ \
         --submitter Biren"
