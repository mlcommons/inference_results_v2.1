# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    gpu_batch_size = 32
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    start_from_device = True
    offline_expected_qps =650
    workspace_size =60000000000     
	
	
@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1_Triton(R5500G5_A100_SXM_80GB_CTSX1):
    use_triton = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    gpu_batch_size = 16
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 2200
    run_infer_on_copy_streams = False
    workspace_size = 35000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True


	
	

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    gpu_batch_size = 16
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
    offline_expected_qps = 2200


