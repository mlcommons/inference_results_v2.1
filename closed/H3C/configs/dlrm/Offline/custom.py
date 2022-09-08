# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    complete_threads = 2
    deque_timeout_usec = 1
    embedding_weights_on_gpu_part = 0.8
    gpu_batch_size = 262100
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 4
    num_staging_threads = 4
    use_jemalloc = True
    offline_expected_qps = 1120000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy(R5300G5_A30X8):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True
    batch_triton_requests = True
    num_concurrent_batchers = 1


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy_Triton(R5300G5_A30X8_HighAccuracy):
    use_triton = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 315000
    offline_expected_qps = 1180000
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    numa_config = "0-1:0-15&2-3:16-31"
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy(R5300G5_A100_PCIE_80GBX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    complete_threads = 1
    deque_timeout_usec = 1
    gemm_plugin_fairshare_cache_size = 18
    gpu_batch_size = 334000
    offline_expected_qps = 1250000 
    max_pairs_per_staging_thread = 262100
    num_staging_batches = 8
    num_staging_threads = 8
    use_jemalloc = True
    numa_config = "0-1:0-31,64-95&2-3:32-63,96-127"
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy(R5300G5_A100_SXM_80GBX4):
    pass


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    gpu_batch_size = 262100
    batch_triton_requests = True
    buffer_manager_thread_count = 0
    gather_kernel_buffer_threshold = 32
    use_triton = True
    num_concurrent_batchers = 1

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy_Triton(R5300G5_A100_SXM_80GBX4_Triton):
    gpu_batch_size = 334000






