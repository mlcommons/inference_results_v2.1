# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIE_80GBx20
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 57500
    soft_drop = 1.0
    gpu_copy_streams = 2
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIE_80GBX20_HighAccuracy(A100_PCIE_80GBX20):
    precision = "fp16"
    server_target_qps = 28000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(ServerGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    active_sms = 60
    gpu_batch_size = 96
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 27000
    deque_timeout_usec = 15000
    run_infer_on_copy_streams = False
    soft_drop = 1
    use_graphs = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM_HighAccuracy(A100_SXM4_80GBX8_CUSTOM):
    precision = "fp16"
    server_target_qps = A100_SXM4_80GBX8_CUSTOM.server_target_qps / 2
    gpu_batch_size = 24
    deque_timeout_usec = 10000

