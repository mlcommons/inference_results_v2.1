# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4
    gpu_inference_streams = 3
    server_target_qps = 136550
    use_cuda_thread_per_device = True
    use_graphs = True        
    



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    server_target_qps = 125000
    use_graphs = False
    use_triton = True
    gpu_batch_size = 32
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    batch_triton_requests = False
    max_queue_delay_usec = 1000



@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 256
    gpu_copy_streams = 4
    gpu_inference_streams = 5
    server_target_qps = 144725
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_Triton(R5300G5_A100_PCIE_80GBX4):
    use_triton = True
    server_target_qps = 50000

	
	

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    active_sms = 100
    input_dtype = "int8"
    input_format = "linear"
    precision = "int8"
    use_deque_limit = True
    deque_timeout_usec = 4000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 141000
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    workspace_size: 7000000000
    numa_config = "0-1:0-31,64-95&2-3:32-63,96-127"
    scenario = Scenario.Server
    benchmark = Benchmark.ResNet50


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    use_triton = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    instance_group_count = 2
    batch_triton_requests = False
    gather_kernel_buffer_threshold = 0
    num_concurrent_batchers = 8
    num_concurrent_issuers = 8
    request_timeout_usec = 2000
    server_target_qps = 118000
