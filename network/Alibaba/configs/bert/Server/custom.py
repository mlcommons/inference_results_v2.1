# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16(A30x8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A30x16
    server_target_qps: int = 23000
    

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_HighAccuracy(SINIAN_VODLA_EFLO_A30x16):
    precision : str = "fp16"
    server_target_qps = 10500
    gpu_inference_streams = 1
    gpu_batch_size = 128

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_Triton(SINIAN_VODLA_EFLO_A30x16):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A30x16_HighAccuracy):
    use_triton = True



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x16
    server_target_qps = 48500
    gpu_batch_size = 256

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_HighAccuracy(SINIAN_VODLA_EFLO_A100x16):
    precision : str = "fp16"
    server_target_qps = 26200

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_Triton(SINIAN_VODLA_EFLO_A100x16):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A100x16_HighAccuracy):
    use_triton = True





@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x32
    server_target_qps = 90000
    gpu_batch_size = 48

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_HighAccuracy(SINIAN_VODLA_EFLO_A100x32):
    precision : str = "fp16"
    server_target_qps = 48000
    gpu_batch_size = 64


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_Triton(SINIAN_VODLA_EFLO_A100x32):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A100x32_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_A30x16(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x16_A30x16
    server_target_qps: int = 62000
    gpu_batch_size = 48
    gpu_inference_streams = 1
    

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_A30x16_HighAccuracy(SINIAN_VODLA_EFLO_A100x16_A30x16):
    precision : str = "fp16"
    server_target_qps = 28000
    gpu_batch_size = 64
    gpu_inference_streams = 2
