# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    multi_stream_expected_latency_ns = 5840000


