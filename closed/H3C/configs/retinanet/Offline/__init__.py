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


ParentConfig = import_module("configs.retinanet")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = False




@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx1
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 540
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_Triton(R5500G5_AMD_A100_SXM_80GBx1):
    use_triton = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx8
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 4400
    run_infer_on_copy_streams = False
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx8_Triton(R5500G5_AMD_A100_SXM_80GBx8):
    start_from_device = None
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1_R4900G5
    gpu_batch_size = 4
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 45


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5_Triton(A2x1_R4900G5):
    use_triton = True
	

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 600
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_Triton(R5500G5_Intel_A100_SXM_80GBx1):
    use_triton = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 4400
    run_infer_on_copy_streams = False
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8_Triton(R5500G5_Intel_A100_SXM_80GBx8):
    start_from_device = None
    use_triton = True

