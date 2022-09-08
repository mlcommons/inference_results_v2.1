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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline

    max_queue_delay_usec = 100




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_Triton(R5500G5_AMD_A100_SXM_80GBx1):
    use_triton = True




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(R5500G5_AMD_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1024
    offline_expected_qps = 10200
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy(A100_PCIe_80GBx3_R4900G5):
    precision = "fp16"
    offline_expected_qps = 4800 



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_Triton(A100_PCIe_80GBx3_R4900G5):
    use_triton = True
    offline_expected_qps = 10000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy_Triton(A100_PCIe_80GBx3_R4900G5_HighAccuracy):
    use_triton = True
    offline_expected_qps = 4500 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1_R4900G5
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 250


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x1_R4900G5_HighAccuracy(A2x1_R4900G5):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 120


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5_Triton(A2x1_R4900G5):
    use_triton = True




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A2x1_R4900G5_HighAccuracy_Triton(A2x1_R4900G5_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 5913 
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy(A30x3_R4900G5):
    precision = "fp16"
    offline_expected_qps = 3042 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5_Triton(A30x3_R4900G5):
    use_triton = True
    offline_expected_qps = 5217



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy_Triton(A30x3_R4900G5_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x3_R4900G5_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 5913 
    workspace_size = 7516192768
    power_limit = 150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A30x3_R4900G5_HighAccuracy_MaxQ(A30x3_R4900G5_MaxQ):
    precision = "fp16"
    offline_expected_qps = 3042 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_HighAccuracy(R5500G5_Intel_A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_Triton(R5500G5_Intel_A100_SXM_80GBx1):
    use_triton = True




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_HighAccuracy_Triton(R5500G5_Intel_A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(R5500G5_Intel_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    offline_expected_qps = 30000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy(R5500G5_Intel_A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_Triton(R5500G5_Intel_A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False




@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000




