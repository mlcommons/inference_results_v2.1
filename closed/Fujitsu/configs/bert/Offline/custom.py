# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class RX2540M6_A30x1_TRT(OfflineGPUBaseConfig):
    system = KnownSystem.RX2540M6_A30x1_TRT
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768
