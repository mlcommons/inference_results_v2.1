# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    use_graphs = False
    run_infer_on_copy_streams = False
    offline_expected_qps = 700000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 43000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    run_infer_on_copy_streams = False
    gpu_batch_size = 2048
    gpu_inference_streams = 2
    gpu_copy_streams = 2
    offline_expected_qps = 360000
    start_from_device = True

