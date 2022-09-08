# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    gpu_batch_size = 1
    offline_expected_qps = 3
    start_from_device = True
    end_on_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1_HighAccuracy(R5500G5_A100_SXM_80GB_CTSX1):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBX20
    gpu_batch_size = 1
    offline_expected_qps = 65

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    gpu_batch_size = 1
    offline_expected_qps = 13


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy(R5300G5_A30X8):
    pass

	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    gpu_batch_size = 1
    offline_expected_qps = 24
    numa_config = "0-3:0-31" 
    start_from_device = True
    end_on_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy(R5300G5_A100_PCIE_80GBX4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    gpu_batch_size = 1
    offline_expected_qps = 10
    numa_config = "0-1:0-31,64-95&2-3:32-63,96-127"
	
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy(R5300G5_A100_SXM_80GBX4):
    pass	
	