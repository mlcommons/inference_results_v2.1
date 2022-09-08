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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    check_contiguity = True
    use_small_tile_gemm_plugin = True



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 405900
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    run_infer_on_copy_streams = True
    offline_expected_qps = 2637000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    gpu_batch_size = 262100
    offline_expected_qps = 2450000
    batch_triton_requests = True
    use_triton = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_Triton):
    gpu_batch_size = 334000
    offline_expected_qps = 2400000



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    complete_threads = 1
    deque_timeout_usec = 1
    gpu_batch_size = 315000
    offline_expected_qps = 855000 
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy(A100_PCIe_80GBx3_R4900G5):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_Triton(A100_PCIe_80GBx3_R4900G5):

    offline_expected_qps = 700000 
    batch_triton_requests = True
    buffer_manager_thread_count = 8 
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy_Triton(A100_PCIe_80GBx3_R4900G5_Triton):

    offline_expected_qps = 700000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 262100
    offline_expected_qps = 420000 
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy(A30x3_R4900G5):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    complete_threads = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 405900
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    run_infer_on_copy_streams = True
    offline_expected_qps = 2637000
    max_pairs_per_staging_thread = 262100
    start_from_device = True
    use_jemalloc = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    gpu_batch_size = 262100
    offline_expected_qps = 2450000
    batch_triton_requests = True
    use_triton = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_Triton):
    gpu_batch_size = 334000
    offline_expected_qps = 2400000




