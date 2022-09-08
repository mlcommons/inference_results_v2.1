# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.bert import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server

    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    active_sms = 60
    gpu_batch_size = 96 
    graphs_max_seqlen = 240
    gpu_copy_streams = 4  
    gpu_inference_streams = 2   
    server_num_issue_query_threads = 0
    server_target_qps = 25900
    run_infer_on_copy_streams = False   
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 12700


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 11205
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 8625 
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy(A100_PCIe_80GBx3_R4900G5):
    precision = "fp16"
    server_target_qps = 4050 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_Triton(A100_PCIe_80GBx3_R4900G5):
    use_triton = True
    server_target_qps = 6750 

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy_Triton(A100_PCIe_80GBx3_R4900G5_HighAccuracy):
    use_triton = True
    server_target_qps = 3562 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0 
    server_target_qps = 4500 
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy(A30x3_R4900G5):
    precision = "fp16"
    server_target_qps = 1950 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5_Triton(A30x3_R4900G5):
    use_triton = True
    server_target_qps = 4200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy_Triton(A30x3_R4900G5_HighAccuracy):
    use_triton = True
    server_target_qps = 1800 
    precision = "fp16"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x3_R4900G5_MaxQ(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 0 
    server_target_qps = 4200 
    soft_drop = 0.993
    power_limit = 150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A30x3_R4900G5_HighAccuracy_MaxQ(A30x3_R4900G5_MaxQ):
    precision = "fp16"
    server_target_qps = 1800 



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy(R5500G5_Intel_A100_SXM_80GBx8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 13100


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_Triton(R5500G5_Intel_A100_SXM_80GBx8):
    server_target_qps = 22400
    use_triton = True




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy):
    gpu_batch_size = 64
    server_target_qps = 11205
    use_triton = True


