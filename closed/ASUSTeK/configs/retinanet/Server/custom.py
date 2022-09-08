# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class ESC4000_A2x8(A2x2):
    system = KnownSystem.ESC4000_A2x8
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 2 
    gpu_inference_streams = 2
    server_target_qps = 330

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
#    gpu_batch_size: int = 0
#    input_dtype: str = ''
#    input_format: str = ''
#    map_path: str = ''
#    precision: str = ''
#    tensor_path: str = ''

    # Optional fields:
#    active_sms: int = 0
#    assume_contiguous: bool = False
#    buffer_manager_thread_count: int = 0
#    cache_file: str = ''
#    complete_threads: int = 0
#    deque_timeout_usec: int = 0
#    gpu_copy_streams: int = 0
#    gpu_inference_streams: int = 0
#    instance_group_count: int = 0
#    model_path: str = ''
#    numa_config: bool = False
#    performance_sample_count_override: int = 0
#    preferred_batch_size: str = ''
#    request_timeout_usec: int = 0
#    run_infer_on_copy_streams: bool = False
#    schedule_rng_seed: int = 0
#    server_num_issue_query_threads: int = 0
#    server_target_latency_ns: int = 0
#    server_target_latency_percentile: float = 0.0
#    server_target_qps: int = 0
#    server_target_qps_adj_factor: float = 0.0
#    use_batcher_thread_per_device: bool = False
#    use_cuda_thread_per_device: bool = False
#    use_deque_limit: bool = False
#    use_graphs: bool = False
#    use_jemalloc: bool = False
#    use_same_context: bool = False
#    use_spin_wait: bool = False
#   warmup_duration: float = 0.0
#   workspace_size: int = 0


