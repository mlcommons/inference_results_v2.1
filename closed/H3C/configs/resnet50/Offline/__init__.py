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
from configs.resnet50 import GPUBaseConfig, CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    run_infer_on_copy_streams = False
    use_graphs = False
    gpu_inference_streams = 1
    gpu_copy_streams = 2


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline






@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx1
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 43000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_Triton(R5500G5_AMD_A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(R5500G5_AMD_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    run_infer_on_copy_streams = False
    gpu_batch_size = 1024    
    gpu_inference_streams = 2
    gpu_copy_streams = 2   
    offline_expected_qps = 350000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    use_triton = True
    offline_expected_qps = 340000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    gpu_batch_size = 2048
    offline_expected_qps = 111000 

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_Triton(A100_PCIe_80GBx3_R4900G5):
    use_triton = True
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1_R4900G5
    gpu_batch_size = 1024
    offline_expected_qps = 3100
    run_infer_on_copy_streams = None


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5_Triton(A2x1_R4900G5):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 2048 
    run_infer_on_copy_streams = True
    offline_expected_qps = 600000 

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5_Triton(A30x3_R4900G5):
    use_triton = True
    offline_expected_qps = 62000 

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x3_R4900G5_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 2048 
    run_infer_on_copy_streams = True
    offline_expected_qps = 600000 
    power_limit = 150



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBX20
    gpu_batch_size = 2048
    offline_expected_qps = 700000
	
	
	



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 45000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_Triton(R5500G5_Intel_A100_SXM_80GBx1):
    start_from_device = None
    gpu_batch_size = 2048
    gpu_copy_streams = 1
    use_triton = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(R5500G5_Intel_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 2
    offline_expected_qps = 350000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_Triton(R5500G5_Intel_A100_SXM_80GBx8):
    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    use_triton = True
    offline_expected_qps = 340000


