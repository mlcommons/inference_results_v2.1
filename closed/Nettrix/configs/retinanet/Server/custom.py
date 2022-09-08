# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 20
    gpu_copy_streams = 2
    gpu_inference_streams = 10
    server_target_qps = 7000
    workspace_size = 160000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 4760
    start_from_device = True
    workspace_size = 70000000000

