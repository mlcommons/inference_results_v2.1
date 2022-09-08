# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class RX2540M6_A30x1_TRT(SingleStreamGPUBaseConfig):
    system = KnownSystem.RX2540M6_A30x1_TRT
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 1126400000
    use_deque_limit: bool = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class RX2540M6_A30x1_TRT_HighAccuracy(RX2540M6_A30x1_TRT):
    pass

