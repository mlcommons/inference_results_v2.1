# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    use_deque_limit = True
    use_cuda_thread_per_device = True
    use_graphs = True
    run_infer_on_copy_streams = False
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 6
    server_target_qps = 610000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    use_deque_limit = True
    deque_timeout_usec = 900
    gpu_batch_size = 128
    server_target_qps = 307300
    start_from_device = True
    use_cuda_thread_per_device = True
    use_graphs = True
    gpu_copy_streams = 3
    gpu_inference_streams = 1

