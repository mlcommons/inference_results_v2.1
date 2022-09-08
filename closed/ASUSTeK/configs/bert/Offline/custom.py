# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000_A2X8(OfflineGPUBaseConfig):
    system = KnownSystem.ESC4000_A2x8

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: int = 0
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    enable_interleaved: bool = False
    gemm_plugin_fairshare_cache_size: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    model_path: str = ''
    numa_config: bool = False
    offline_expected_qps: int = 0
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    soft_drop: float = 0.0
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_small_tile_gemm_plugin: bool = False
    use_spin_wait: bool = False
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000_A2X8_HighAccuracy(ESC4000_A2X8):
    precision = "fp16"
    offline_expected_qps = ESC4000_A2X8.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000_A2X8_Triton(ESC4000_A2X8):
    use_triton = True

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    gpu_batch_size: int = 0
    input_dtype: str = ''
    input_format: str = ''
    precision: str = ''
    tensor_path: str = ''

    # Optional fields:
    active_sms: int = 0
    batch_triton_requests: bool = False
    bert_opt_seqlen: int = 0
    buffer_manager_thread_count: int = 0
    cache_file: str = ''
    coalesced_tensor: bool = False
    deque_timeout_usec: int = 0
    enable_interleaved: bool = False
    gather_kernel_buffer_threshold: int = 0
    gemm_plugin_fairshare_cache_size: int = 0
    gpu_copy_streams: int = 0
    gpu_inference_streams: int = 0
    graph_specs: str = ''
    graphs_max_seqlen: int = 0
    instance_group_count: int = 0
    max_queue_delay_usec: int = 0
    model_path: str = ''
    num_concurrent_batchers: int = 0
    num_concurrent_issuers: int = 0
    numa_config: bool = False
    offline_expected_qps: int = 0
    output_pinned_memory: bool = False
    performance_sample_count_override: int = 0
    preferred_batch_size: str = ''
    request_timeout_usec: int = 0
    run_infer_on_copy_streams: bool = False
    soft_drop: float = 0.0
    use_concurrent_harness: bool = False
    use_graphs: bool = False
    use_jemalloc: bool = False
    use_small_tile_gemm_plugin: bool = False
    use_spin_wait: bool = False
    workspace_size: int = 0


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ESC4000_A2X8_HighAccuracy_Triton(ESC4000_A2X8_HighAccuracy):
    use_triton = True


