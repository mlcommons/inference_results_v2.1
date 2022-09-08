import os
import sys
sys.path.insert(0, os.getcwd())

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.3d-unet")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    gpu_inference_streams = 1
    gpu_copy_streams = 1


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class NV72ads_A10_v5(OfflineGPUBaseConfig):
    system = KnownSystem.NV72ads_A10_v5
    gpu_batch_size = 2
    offline_expected_qps = 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class NV72ads_A10_v5_HighAccuracy(NV72ads_A10_v5):
    pass
