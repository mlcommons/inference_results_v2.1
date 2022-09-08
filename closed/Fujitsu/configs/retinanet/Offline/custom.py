# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class RX2540M6_A30x1_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.RX2540M6_A30x1_TRT
    gpu_batch_size = 8
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 280
    run_infer_on_copy_streams = False
    workspace_size = 70000000000


