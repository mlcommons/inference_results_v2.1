# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1(SingleStreamGPUBaseConfig):
    system = KnownSystem.R5500G5_A100_SXM_80GB_CTSx1

    # Applicable fields for this benchmark are listed below. Not all of these are necessary, and some may be defined in the BaseConfig already and inherited.
    # Please see NVIDIA's submission config files for example values and which fields to keep.
    # Required fields (Must be set or inherited to run):
    enable_interleaved = False
    single_stream_expected_latency_ns =1700000 


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5500G5_A100_SXM_80GB_CTSX1_Triton(R5500G5_A100_SXM_80GB_CTSX1):
    use_triton = True


