# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class RX2540M6_A30x1_TRT(SingleStreamGPUBaseConfig):
    system = KnownSystem.RX2540M6_A30x1_TRT
    single_stream_expected_latency_ns = 7400000

