# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(MultiStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    start_from_device = True
    multi_stream_expected_latency_ns = 693000

