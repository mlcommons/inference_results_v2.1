# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    single_stream_expected_latency_ns = 730000
    disable_beta1_smallk = True


