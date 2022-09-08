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


GPUBaseConfig = import_module("configs.retinanet").GPUBaseConfig


class MultiStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.MultiStream
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    multi_stream_samples_per_query = 8
    multi_stream_target_latency_percentile = 99
    use_graphs = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1(MultiStreamGPUBaseConfig):
    system = KnownSystem.R5500G5_AMD_A100_SXM_80GBx1
    start_from_device = True
    multi_stream_expected_latency_ns = 9000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_AMD_A100_SXM_80GBx1_Triton(R5500G5_AMD_A100_SXM_80GBx1):
    use_triton = True



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5(MultiStreamGPUBaseConfig):
    system = KnownSystem.A2x1_R4900G5
    multi_stream_expected_latency_ns = 114000000
    min_duration = 3600000
    min_query_count = 16384


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5_Triton(A2x1_R4900G5):
    use_triton = True
    multi_stream_expected_latency_ns = 114100000

	
@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(MultiStreamGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    start_from_device = True
    multi_stream_expected_latency_ns = 7000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1_Triton(R5500G5_Intel_A100_SXM_80GBx1):
    use_triton = True

