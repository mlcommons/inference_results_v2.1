# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(OfflineGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1
    gpu_batch_size =1024   
    gpu_copy_streams =2   
    run_infer_on_copy_streams = True
    start_from_device = True
    offline_expected_qps = 44000 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1_Triton(R5500G5_A100_SXM_80GB_CTSX1):
    use_triton = True
    start_from_device = None
    gpu_batch_size =2048
    gpu_copy_streams = 1      


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(OfflineGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBX20
    gpu_batch_size = 2048
    offline_expected_qps = 700000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    run_infer_on_copy_streams = True
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    offline_expected_qps = 150000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True
    offline_expected_qps = 161200.0


	

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    gpu_batch_size = 2048
    offline_expected_qps = 149500


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_Triton(R5300G5_A100_PCIE_80GBX4):
    use_triton = True
    offline_expected_qps = 150000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4

    gpu_inference_streams = 2
    use_graphs = False
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    workspace_size = 7000000000
    start_from_device = True
    run_infer_on_copy_streams = False
    numa_config = "0-1:0-31,64-95&2-3:32-63,96-127"
    scenario = Scenario.Offline
    benchmark = Benchmark.ResNet50
    offline_expected_qps = 170000 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    use_triton = True

    gpu_copy_streams = 1
    num_concurrent_batchers = 6
    num_concurrent_issuers = 6
    instance_group_count = 2
    batch_triton_requests = True
    offline_expected_qps = 160664