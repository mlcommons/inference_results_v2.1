# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    #offline_expected_qps = 13600
    offline_expected_qps = 15000
    workspace_size = 7516192768
    start_from_device=True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_A100_PCIE_80GBx4_HighAccuracy(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 7000
    workspace_size = 7516192768
    start_from_device=True
    precision = "fp16"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 16000
    start_from_device = True
    workspace_size = 7516192768
    numa_config: "0-15,128-143&1:16-31,144-159&32-47,160-175&0:48-63,176-191&64-79,192-207&3:80-95,208-223&96-111,224-239&2:112-127,240-255&112-127,240-255"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8545_A100_SXM_80GBX4_HighAccuracy(XE8545_A100_SXM_80GBX4):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 8000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XE8545_A100_SXM_80GBX4_MAXQ(OfflineGPUBaseConfig):
    system = KnownSystem.XE8545_A100_SXM_80GBx4

    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 20000
    workspace_size = 7516192768
    power_limit = 275

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XE8545_A100_SXM_80GBX4_MAXQ_HighAccuracy(XE8545_A100_SXM_80GBX4_MAXQ):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = XE8545_A100_SXM_80GBX4_MAXQ.offline_expected_qps / 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR12_A2x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 250
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR12_A2x1_HighAccuracy_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR12_A2x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 125
    start_from_device = True
    precision = "fp16"

