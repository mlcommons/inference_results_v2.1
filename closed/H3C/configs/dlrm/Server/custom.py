# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_num_bundles = 2
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 131000
    num_staging_batches = 8
    num_staging_threads = 8
    server_target_qps = 800000
    use_jemalloc = False
    numa_config = "0-7:0-31"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy(R5300G5_A30X8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True
    server_target_qps = 650000
    batch_triton_requests = True
    gather_kernel_buffer_threshold = 64



@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy_Triton(R5300G5_A30X8_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 1145000
    use_jemalloc = False
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    start_from_device = True
    numa_config = "0-3:0-31"


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy(R5300G5_A100_PCIE_80GBX4):
    pass
    server_target_qps = 1152000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    deque_timeout_usec = 1
    gpu_batch_size = 274000
    gpu_num_bundles = 2
    num_staging_batches = 4
    num_staging_threads = 4
    server_target_qps = 950000
    use_jemalloc = True
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 18
    numa_config = "0-1:0-31,64-95&2-3:32-63,96-127"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy(R5300G5_A100_SXM_80GBX4):
    pass

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    server_target_qps = 400000
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    max_queue_delay_usec = 1
    use_triton = True
    gather_kernel_buffer_threshold = 10


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy_Triton(R5300G5_A100_SXM_80GBX4_Triton):
    pass
