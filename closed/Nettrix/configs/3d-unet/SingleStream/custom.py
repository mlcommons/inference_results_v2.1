# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(SingleStreamGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    gpu_batch_size = 1
    start_from_device = True
    end_on_device = True
    single_stream_expected_latency_ns = 552000000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM_HighAccuracy(A100_SXM4_80GBX1_CUSTOM):
    pass

