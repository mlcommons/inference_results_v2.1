# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16(A30x8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A30x16
    offline_expected_qps = 26000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_HighAccuracy(SINIAN_VODLA_EFLO_A30x16):
    precision : str = "fp16"
    offline_expected_qps = 16000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_Triton(SINIAN_VODLA_EFLO_A30x16):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A30x16_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A30x16_HighAccuracy):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x16
    offline_expected_qps = 60000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_HighAccuracy(SINIAN_VODLA_EFLO_A100x16):
    precision : str = "fp16"
    offline_expected_qps = 30000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_Triton(SINIAN_VODLA_EFLO_A100x16):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A100x16_HighAccuracy):
    use_triton = True



@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x32
    offline_expected_qps = 120000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_HighAccuracy(SINIAN_VODLA_EFLO_A100x32):
    precision : str = "fp16"
    offline_expected_qps = 60000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_Triton(SINIAN_VODLA_EFLO_A100x32):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x32_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A100x32_HighAccuracy):
    use_triton = True




@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x8_A30x8(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x8_A30x8
    offline_expected_qps = 52000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x8_A30x8_HighAccuracy(SINIAN_VODLA_EFLO_A100x8_A30x8):
    precision : str = "fp16"
    offline_expected_qps = 3000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x8_A30x8_Triton(SINIAN_VODLA_EFLO_A100x8_A30x8):
    use_triton = True

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x8_A30x8_HighAccuracy_Triton(SINIAN_VODLA_EFLO_A100x8_A30x8_HighAccuracy):
    use_triton = True

#SINIAN_VODLA_EFLO_A100x16_A30x16
@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_A30x16(A100_SXM_80GBx8):
    system = KnownSystem.SINIAN_VODLA_EFLO_A100x16_A30x16
    offline_expected_qps = 104000
    gpu_batch_size = 1024
    gpu_inference_streams = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class SINIAN_VODLA_EFLO_A100x16_A30x16_HighAccuracy(SINIAN_VODLA_EFLO_A100x16_A30x16):
    precision : str = "fp16"
    offline_expected_qps = 50000
    gpu_batch_size = 1024

