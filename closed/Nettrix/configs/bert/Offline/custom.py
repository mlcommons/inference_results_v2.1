# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 72000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIE_80GBX20_HighAccuracy(A100_PCIE_80GBX20):
    precision = "fp16"
    offline_expected_qps = 38000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM_HighAccuracy(A100_SXM4_80GBX1_CUSTOM):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 2
    offline_expected_qps = 31000
    workspace_size = 7516192768
    gpu_copy_streams = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM_HighAccuracy(A100_SXM4_80GBX8_CUSTOM):
    precision = "fp16"
    offline_expected_qps = A100_SXM4_80GBX8_CUSTOM.offline_expected_qps / 2

