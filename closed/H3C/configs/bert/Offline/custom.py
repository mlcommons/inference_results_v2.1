# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size =1280    
    start_from_device = True
    gpu_inference_streams =1 
    offline_expected_qps =3500  

	
	
	
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1_Triton(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500
    use_triton = True

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X5_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x5_R4900G5
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 1250
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBX20
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768
    offline_expected_qps = 68000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIE_80GBX20_HighAccuracy(A100_PCIE_80GBX20):
    precision = "fp16"
    offline_expected_qps = A100_PCIE_80GBX20.offline_expected_qps / 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 13000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy(R5300G5_A30X8):
    precision = "fp16"
    offline_expected_qps = 8119.999999999999


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy_Triton(R5300G5_A30X8_HighAccuracy):
    use_triton = True

	
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    gpu_batch_size = 1024
    offline_expected_qps = 15500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy(R5300G5_A100_PCIE_80GBX4):
    precision = "fp16"
    offline_expected_qps = 6300


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_Triton(R5300G5_A100_PCIE_80GBX4):
    use_triton = True

    offline_expected_qps = 15550


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy_Triton(R5300G5_A100_PCIE_80GBX4_HighAccuracy):
    use_triton = True
    offline_expected_qps = 6100


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    use_small_tile_gemm_plugin = True
    gemm_plugin_fairshare_cache_size = 120
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy(R5300G5_A100_SXM_80GBX4):
    precision = "fp16"
    offline_expected_qps = 7500
    gpu_batch_size = 512


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy_Triton(R5300G5_A100_SXM_80GBX4_HighAccuracy):
    use_triton = True
