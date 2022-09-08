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


GPUBaseConfig = import_module("configs.ssd-mobilenet").GPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = False


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBx1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 49000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_ARMx1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 49000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_MIG_1x1g_5gb
    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    workspace_size = 2147483648
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GB_MIG_1x1g_10gb
    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    workspace_size = 2147483648
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    offline_expected_qps = 6600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_40GB_ARMx1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 44000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb
    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    start_from_device = True
    workspace_size = 2147483648
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx1
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    start_from_device = True
    offline_expected_qps = 51200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.TritonUnified, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1_Triton):
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GBx8
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    start_from_device = True
    offline_expected_qps = 409600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    start_from_device = None
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARM_MIG_1x1g_10gb
    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    workspace_size = 2147483648
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx1
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    offline_expected_qps = 55000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_80GB_ARMx8
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    offline_expected_qps = 335650


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_40GB_MIG_1x1g_5gb
    gpu_inference_streams = 1
    gpu_batch_size = 256
    gpu_copy_streams = 2
    start_from_device = True
    workspace_size = 2147483648
    offline_expected_qps = 7000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GB_MIG_1x1g5gb_Triton(A100_SXM_40GB_MIG_1x1g5gb):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GBx1(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_40GBx1
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    start_from_device = True
    offline_expected_qps = 51200


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GBx1_Triton(A100_SXM_40GBx1):
    start_from_device = None
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM_40GBx8
    gpu_inference_streams = 1
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    start_from_device = True
    offline_expected_qps = 409600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM_40GBx8_Triton(A100_SXM_40GBx8):
    start_from_device = None
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1(OfflineGPUBaseConfig):
    system = KnownSystem.A2x1
    gpu_inference_streams = 1
    gpu_batch_size = 768
    gpu_copy_streams = 2
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    offline_expected_qps = 4600


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2x1_Triton(A2x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    system = KnownSystem.A30_MIG_1x1g_6gb
    gpu_inference_streams = 1
    gpu_batch_size = 128
    gpu_copy_streams = 2
    run_infer_on_copy_streams = False
    workspace_size = 370561024
    offline_expected_qps = 7050


@ConfigRegistry.register(HarnessType.HeteroMIG, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 5782


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True
    offline_expected_qps = 6800


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1(OfflineGPUBaseConfig):
    system = KnownSystem.A30x1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    gpu_copy_streams = 4
    run_infer_on_copy_streams = False
    offline_expected_qps = 26000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class A30x1_Triton(A30x1):
    gpu_copy_streams = 1
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1(OfflineGPUBaseConfig):
    system = KnownSystem.T4x1
    gpu_inference_streams = 1
    gpu_batch_size = 128
    gpu_copy_streams = 4
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    offline_expected_qps = 7463


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x1_Triton(T4x1):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20(OfflineGPUBaseConfig):
    system = KnownSystem.T4x20
    gpu_inference_streams = 1
    gpu_batch_size = 128
    gpu_copy_streams = 4
    offline_expected_qps = 152500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x20_Triton(T4x20):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8(OfflineGPUBaseConfig):
    system = KnownSystem.T4x8
    gpu_inference_streams = 1
    gpu_batch_size = 128
    gpu_copy_streams = 4
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"
    offline_expected_qps = 62800


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class T4x8_Triton(T4x8):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin(OfflineGPUBaseConfig):
    system = KnownSystem.Orin
    # GPU-only QPS
    _gpu_offline_expected_qps = 5500
    # DLA-only (1x) QPS
    _dla_offline_expected_qps = 800
    # GPU + 2 DLA QPS
    offline_expected_qps = 7100
    use_direct_host_access = True
    gpu_inference_streams = 1
    dla_batch_size = 64
    dla_copy_streams = 1
    dla_inference_streams = 1
    dla_core = 0
    gpu_batch_size = 128
    gpu_copy_streams = 1
    input_format = "chw4"
    tensor_path = "build/preprocessed_data/coco/val2017/SSDMobileNet/int8_chw4"


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class Orin_MaxQ(Orin):
    soc_cpu_freq = 1036800
    soc_gpu_freq = 828750000
    soc_dla_freq = 960000000
    soc_emc_freq = 3199000000
    orin_num_cores = 4
    offline_expected_qps = 5331.94
