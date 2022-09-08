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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100
    use_graphs = False
    use_cuda_thread_per_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 4000
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 2300
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 2300
    power_limit = 200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_Triton):
    server_target_qps = 2100
    power_limit = 200


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 4100
    start_from_device = True
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    start_from_device = None
    server_target_qps = 4000
    instance_group_count = 4
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 2500
    power_limit = 250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_Triton):
    server_target_qps = 2400
    power_limit = 250


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2(ServerGPUBaseConfig):
    system = KnownSystem.A2x2
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 60


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x2_Triton(A2x2):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NF5468M6J_A100_PCIe_80GBx24(ServerGPUBaseConfig):
    system = KnownSystem.NF5468M6J_A100_PCIe_80GBx24
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 2
    server_target_qps = 11000
    run_infer_on_copy_streams = False
    use_graphs = True
    numa_config = "0,1,2,3,4,5,6,7,8,9,10,11:0-39,80-119&12,13,14,15,16,17,18,19,20,21,22,23:40-79,120-159"
    workspace_size = 70000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NF5688M6_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.NF5688M6_A100_SXM_80GBx8
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 24
    gpu_inference_streams = 3
    server_target_qps = 4650
    start_from_device = True
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class NF5488A5_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.NF5488A5_A100_SXM_80GBx8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 4100
    start_from_device = True
    workspace_size = 70000000000
