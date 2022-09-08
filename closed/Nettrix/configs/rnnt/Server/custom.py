# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    dali_pipeline_depth = 1
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    run_infer_on_copy_streams = False
    server_num_issue_query_threads = 0
    server_target_qps = 112000
    start_from_device = True

