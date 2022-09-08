# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX1_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBX1_CUSTOM
    gpu_batch_size = 2048
    offline_expected_qps = 13800
    start_from_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class A100_SXM4_80GBX8_CUSTOM(OfflineGPUBaseConfig):
    system = KnownSystem.A100_SXM4_80GBx8_Custom
    audio_batch_size = 768
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 120000
    num_warmups = 40480
    nobatch_sorting = True
    gpu_batch_size = 2048
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    run_infer_on_copy_streams = False
    start_from_device = True

