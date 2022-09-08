# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE8545_A100_SXM_80GBX4_MAXQ(ServerGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    active_sms = 87
    gpu_batch_size = 80
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 10500
    soft_drop = 0.9921656203806923
    power_limit = 275
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE8545_A100_SXM_80GBX4_MAXQ_HighAccuracy(XE8545_A100_SXM_80GBX4_MAXQ):
    precision = "fp16"
    active_sms = 32
    gpu_batch_size = 95
    server_target_qps = 5300
    power_limit = 300
    soft_drop = 0.9912248225113302



