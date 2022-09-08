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

MLPERF_TEST_MODE=`sed -n ${TC_NUM}p test_case.conf | awk '{print $2}'`
MODE=`sed -n ${TC_NUM}p test_case.conf | awk '{print $3}'`
SYSTEM_ID=`sed -n ${TC_NUM}p test_case.conf | awk '{print $4}'`
# echo $MLPERF_TEST_MODE
# echo $MODE
# echo $SYSTEM_ID

BASH_NAME=$1
exec_shell="${BASH_NAME} ${MLPERF_TEST_MODE} ${MODE} ${SYSTEM_ID}"
bash $exec_shell


