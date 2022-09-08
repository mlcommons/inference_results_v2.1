# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py
import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.bert import GPUBaseConfig, CPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server

    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A2X5_R4900G5(ServerGPUBaseConfig):
    system = KnownSystem.A2x5_R4900G5
    enable_interleaved = False
    gpu_batch_size = 2  
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 800
    soft_drop = 0.993



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_PCIE_80GBX20(ServerGPUBaseConfig):
    system = KnownSystem.A100_PCIe_80GBX20
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 60000.0
    soft_drop = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class A100_PCIE_80GBX20_HighAccuracy(A100_PCIE_80GBX20):
    precision = "fp16"
    server_target_qps = A100_PCIE_80GBX20.server_target_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A30x8
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12370
    soft_drop = 0.993


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy(R5300G5_A30X8):
    precision = "fp16"
    server_target_qps = 5250


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A30X8_Triton(R5300G5_A30X8):
    use_triton = True
    server_target_qps = 12500
    max_queue_delay_usec = 1000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A30X8_HighAccuracy_Triton(R5300G5_A30X8_HighAccuracy):
    use_triton = True
    server_target_qps = 5850

	
	
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_PCIE_80GBX4

    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12040
    soft_drop = 1.0
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy(R5300G5_A100_PCIE_80GBX4):
    precision = "fp16"
    server_target_qps = 5750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_Triton(R5300G5_A100_PCIE_80GBX4):
    use_triton = True

    server_target_qps = 11100
    start_from_device = False


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_PCIE_80GBX4_HighAccuracy_Triton(R5300G5_A100_PCIE_80GBX4_HighAccuracy):
    use_triton = True
    start_from_device = False
    server_target_qps = 5750


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R5300G5_A100_SXM_80GBx4
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 12000 
    start_from_device = True
    soft_drop = 1.0
    gpu_copy_streams = 2
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy(R5300G5_A100_SXM_80GBX4):
    precision = "fp16"
    server_target_qps = 4860
    gpu_copy_streams = 1
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_Triton(R5300G5_A100_SXM_80GBX4):
    gpu_inference_streams = None
    instance_group_count = 2
    server_target_qps = 10800
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G5_A100_SXM_80GBX4_HighAccuracy_Triton(R5300G5_A100_SXM_80GBX4_HighAccuracy):
    server_target_qps = 4650
    use_triton = True




