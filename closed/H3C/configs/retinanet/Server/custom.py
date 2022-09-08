# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 2230
    workspace_size = 40000000000
    

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    server_target_qps = 2210
    instance_group_count = 2
    use_triton = True
	
	

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 2000
    workspace_size = 70000000000

