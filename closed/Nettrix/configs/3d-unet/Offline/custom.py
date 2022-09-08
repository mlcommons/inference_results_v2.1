# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    gpu_batch_size = 1
    offline_expected_qps = 64
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    numa_config = "0-9:0-15,32-47&10-19:16-31,48-63"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIE_80GBX20_HighAccuracy(A100_PCIE_80GBX20):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    gpu_batch_size = 1
    offline_expected_qps = 3
    start_from_device = True
    end_on_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM_HighAccuracy(A100_SXM4_80GBX1_CUSTOM):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    gpu_batch_size = 1
    start_from_device = True
    end_on_device = True
    offline_expected_qps = 28
    gpu_inference_streams = 1
    gpu_copy_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM_HighAccuracy(A100_SXM4_80GBX8_CUSTOM):
    offline_expected_qps = 28

