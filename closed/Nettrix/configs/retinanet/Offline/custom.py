# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    gpu_batch_size = 66
    gpu_copy_streams = 1
    gpu_inference_streams = 20
    offline_expected_qps = 11000
    run_infer_on_copy_streams = False
    workspace_size = 160000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps = 650
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    gpu_batch_size = 32
    gpu_copy_streams = 1
    gpu_inference_streams = 4
    start_from_device = True
    offline_expected_qps = 5000
    run_infer_on_copy_streams = False
    workspace_size = 70000000000

