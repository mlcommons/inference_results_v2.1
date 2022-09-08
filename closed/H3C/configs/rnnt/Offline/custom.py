# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    gpu_batch_size = 2048
    offline_expected_qps = 13800
    start_from_device = True
    gpu_copy_streams =2 


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X5_R4900G5(OfflineGPUBaseConfig):
    system = KnownSystem.A2x5_R4900G5
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 6000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    gpu_batch_size = 2048
    offline_expected_qps = 55679.99999999999

	
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    gpu_batch_size = 2048
    offline_expected_qps = 56000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 48000
    gpu_copy_streams= 2 