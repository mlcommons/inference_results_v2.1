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
from configs.rnnt import GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = True
    num_warmups = 512
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 1
    gpu_copy_streams = 1




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    gpu_batch_size = 2048
    offline_expected_qps = 13800
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(R5500G5_Intel_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 105500
    num_warmups = 40480
    nobatch_sorting = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx3_R4900G5
    gpu_batch_size = 2048
    offline_expected_qps = 39900


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1_R4900G5
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1150

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x3_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 2048
    offline_expected_qps = 20200 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class A30x3_R4900G5_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.A30x3_R4900G5
    gpu_batch_size = 2048
    offline_expected_qps = 20200 
    power_limit = 150


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx1
    gpu_batch_size = 2048      
    offline_expected_qps = 13800  
    start_from_device = True
    gpu_copy_streams = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_Intel_A100_SXM_80GBx8(R5500G5_Intel_A100_SXM_80GBx1):
    system = KnownSystem.R5500G5_Intel_A100_SXM_80GBx8
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 105000
    num_warmups = 40480
    nobatch_sorting = True





