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

arg_cnt=$#

if [ $arg_cnt -lt 2 ]; then
  echo "Use command: bert_99.9_compliance.sh systemids scenarios testids" 
  echo "i.e.: bert_99.9_compliance_run.sh BR104-300W-PCIex8,BR104-300W-PCIex4 Offline,Server TEST01,TEST05"
  echo "  or: bert_99.9_compliance_run.sh BR104-300W-PCIex8,BR104-300W-PCIex4 Offline,Server"
  exit 1
fi

all_systemids=$1
all_systemids=${all_systemids//,/ }
all_scenarios=$2
all_scenarios=${all_scenarios//,/ }

all_tests="TEST01 TEST05"
if [[ $arg_cnt -ge 3 ]]; then
  all_tests=$3
  all_tests=${all_tests//,/ }
fi

echo "Run test for system: " $all_systemids
echo "Run test for system: " $all_scenarios
echo "Run test for system: " $all_tests

function run_cmd()
{
  local cmdline=$*
  echo "run command: "${cmdline}
  ${cmdline}
}

function run_compliance_TESTID()
{
  local scenario=$1
  local systemid=$2
  local testid=$3
  echo ${scenario} ${systemid} ${testid}

  cmdline="./bert_99.9_compliance.sh perf ${scenario} ${systemid} ${testid}"
  run_cmd ${cmdline}

  run_cmd "sleep 3m"
}

function run_test_scenario()
{
  local scenario=$1
  local systemid=$2

  cmdline="./bert_99.9.sh accu ${scenario} ${systemid}"
  run_cmd ${cmdline}

  # run_cmd "sleep 1m"

  cmdline="./bert_99.9.sh perf ${scenario} ${systemid}"
  run_cmd ${cmdline}

  # run_cmd "sleep 3m"

  for testid in TEST01 TEST05
  do
    echo ${testid}
    run_compliance_TESTID ${scenario} ${systemid} ${testid}
  done
}

function run_test_group()
{
  local systemid=$1
  # for scenario in Offline Server
  for scenario in Server
  do
    echo ${scenario}
    run_test_scenario ${scenario} ${systemid}
  done
}

# for systemid in BR104-300W-PCIex8 BR104-300W-PCIex4 BR104-300W-PCIex2 BR104-300W-PCIex1
for systemid in $all_systemids
do
  echo ${systemid}
  for scenario in $all_scenarios
  do
    echo "Start to run test for " ${scenario}  ${systemid}
    run_test_scenario ${scenario} ${systemid}
  done
done
