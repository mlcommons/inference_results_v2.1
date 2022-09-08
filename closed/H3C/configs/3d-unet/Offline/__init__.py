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

ParentConfig = import_module("configs.3d-unet")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 1


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx1
    gpu_batch_size = 1
    offline_expected_qps = 3
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_Triton(R5500G5_AMD_A100_SXM_80GBx1):
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(R5500G5_AMD_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    offline_expected_qps = 24


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy(R5500G5_AMD_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    use_graphs = True
    instance_group_count = 4
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_AMD_A100_SXM_80GBx8_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    gpu_batch_size = 1
    offline_expected_qps = 7.8 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5_HighAccuracy(A100_PCIe_80GBx3_R4900G5):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 1
    offline_expected_qps = 4.9 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A30x3_R4900G5_HighAccuracy(A30x3_R4900G5):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x3_R4900G5_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 1
    offline_expected_qps = 4.9
    power_limit = 150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class A30x3_R4900G5_HighAccuracy_MaxQ(A30x3_R4900G5_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    gpu_batch_size = 1
    offline_expected_qps = 3
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_HighAccuracy(R5500G5_Intel_A100_SXM_80GBx1):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_Triton(R5500G5_Intel_A100_SXM_80GBx1):
    instance_group_count = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_HighAccuracy_Triton(R5500G5_Intel_A100_SXM_80GBx1_Triton):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(R5500G5_Intel_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    offline_expected_qps = 24


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy(R5500G5_Intel_A100_SXM_80GBx8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_Triton(R5500G5_Intel_A100_SXM_80GBx8):
    use_graphs = True
    instance_group_count = 4
    use_triton = True
    output_pinned_memory = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_HighAccuracy_Triton(R5500G5_Intel_A100_SXM_80GBx8_Triton):
    pass



