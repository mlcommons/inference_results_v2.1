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

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.dlrm")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    complete_threads = 1



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2450000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    deque_timeout_usec = 1
    gpu_batch_size = 224000 
    gpu_num_bundles = 2
    num_staging_batches = 2 
    num_staging_threads = 4
    server_target_qps = 2460 
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    embedding_weights_on_gpu_part = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy(A100_PCIe_80GBx3_R4900G5):
    pass



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_Triton(A100_PCIe_80GBx3_R4900G5):

    server_target_qps =200000 
    batch_triton_requests = True

    use_triton = True
    gather_kernel_buffer_threshold = 64
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy_Triton(A100_PCIe_80GBx3_R4900G5_Triton):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 131000 
    gpu_num_bundles = 2
    num_staging_batches = 8 
    num_staging_threads = 8 
    server_target_qps = 400000 
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy(A30x3_R4900G5):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 2450000
    start_from_device = True
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(ServerGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    gpu_batch_size = 262100
    gpu_num_bundles = 2
    server_target_qps = 725000
    use_jemalloc = True
    max_queue_delay_usec = 1000
    use_triton = True
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64
    request_timeout_usec = 2000
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_Triton):
    pass




