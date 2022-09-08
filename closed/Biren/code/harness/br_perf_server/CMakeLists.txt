# 
# Copyright (c) 2022, Biren Technologies.  All rights reserved.
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

set(UMD_PATH /home/br104/umd/output)
set(UMD_INC ${UMD_PATH}/include)
set(UMD_LIB ${UMD_PATH}/lib)
set(SUDA_LIB /usr/local/lib)
find_library(UMD_LIB_BESU NAMES besu HINTS ${UMD_LIB})
find_library(SUDA NAMES brsuda HINTS ${SUDA_LIB})

add_library(br_perf_server
    src/br_perf_server.cpp
    src/timer.cpp
)

target_link_libraries(br_perf_server
    glog
    ${SUDA}
    ${UMD_LIB_BESU}
)

target_include_directories(br_perf_server
    PUBLIC
        ${LOADGEN_INCLUDE_DIR}
        ${UMD_INC}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../common>
)
