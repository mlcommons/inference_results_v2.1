# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Any, Callable, Dict, Final, Optional, List, Union

import os
import sys
import importlib.util

from code.common import logging
from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


# Dynamically build Enum for known systems
_system_confs = dict()


def add_systems(name_format_string: str,
                id_format_string: str,
                cpu: KnownCPU,
                accelerator: KnownGPU,
                accelerator_counts: List[int],
                mem_requirement: Memory,
                target_dict: Dict[str, SystemConfiguration] = _system_confs):
    """Adds a SystemConfiguration to a dictionary.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
        target_dict (Dict[str, SystemConfiguration]): The dictionary to add the SystemConfiguration to.
                                                      (Default: _system_confs)
    """
    for count in accelerator_counts:
        target_dict[name_format_string.format(count)] = SystemConfiguration(
            CPUConfiguration({cpu: MATCH_ANY}),
            min_memory_requirement(mem_requirement),
            AcceleratorConfiguration({accelerator: count}),
            numa_conf=MATCH_ANY,
            system_id=id_format_string.format(count))

# Hopper systems
# TODO: Placeholder for H100-pcie system

# A100_PCIe_40GB and 80GB based systems:
add_systems("A100_PCIe_80GBx{}", "A100-PCIe-80GBx{}",
            MatchAllowList([KnownCPU.AMD_EPYC_7742.value, KnownCPU.x86_64_Generic.value]),
            KnownGPU.A100_PCIe_80GB.value, [8], Memory(30, ByteSuffix.GiB))
# FIXME: not sure what's causing ipp1-1468's host memory to be less than 1 TiB, so using 1TB for now

# A100_SXM4_40GB and SXM_80GB based systems:

# Other Ampere based systems

# Turing based systems

# Embedded systems

# Intel CPU-based systems (no discrete accelerator)

# Inferentia-based system

"""
Handle custom systems to better support partner drops. The custom_list by default is located at
code.common.systems.custom_list, but for testing and developer use, you can set the environment variable
MLPINF_CUSTOM_DEFINITION_PATH to look in a different directory.
This expects the directory at this path to contain:
    - custom_systems/custom_list.py
    - custom_configs/<benchmark>/<scenario>/custom.py
"""
_custom_definition_path = os.environ.get("MLPINF_CUSTOM_DEFINITION_PATH", None)
if _custom_definition_path is None:
    if importlib.util.find_spec("code.common.systems.custom_list") is not None:
        from code.common.systems.custom_list import custom_systems
        _system_confs.update(custom_systems)
elif not os.path.exists(_custom_definition_path):
    raise FileNotFoundError(f"MLPINF_CUSTOM_DEFINITION_PATH {_custom_definition_path} does not exist.")
else:
    from code.common.fix_sys_path import ScopedRestrictedImport
    with ScopedRestrictedImport(restricted_path=[_custom_definition_path] + sys.path) as sri:
        if importlib.util.find_spec("custom_systems.custom_list") is not None:
            from custom_systems.custom_list import custom_systems
            _system_confs.update(custom_systems)
KnownSystem = MatchableEnum("KnownSystem", _system_confs)


_deprecated_systems = dict()
DeprecatedSystem = MatchableEnum("DeprecatedSystem", _deprecated_systems)


def match_known_system(sys_conf):
    """Matches a SystemConfiguration with KnownSystems and returns the enum member sys_conf matched with. Also sets the
    system_id field of sys_conf to the system_id of the enum member.

    Returns None if no match was found."""
    match = KnownSystem.get_first_match(sys_conf)
    if match is None:
        # Check if the system is a deprecated system. If so, log a warning. It is up to the caller to throw the error
        # (main.py will throw an error, since we still return None for MATCHED_SYSTEM).
        match = DeprecatedSystem.get_first_match(sys_conf)
        if match is not None:
            logging.warn(f"Detected system is a deprecated system, and is no longer supported: {match}")
        return None
    sys_conf.set_id(match.value.system_id)
    return match


DETECTED_SYSTEM = SystemConfiguration.detect()
"""SystemConfiguration: The detected SystemConfiguration of the current system at runtime"""

MATCHED_SYSTEM = match_known_system(DETECTED_SYSTEM)
"""KnownSystem: The KnownSystem Enum member that DETECTED_SYSTEM matched with. Used to select BenchmarkConfigurations."""


def _default_on_matched(f):
    """Decorator that takes in a classmethod Callable that takes in a System as a parameter, and returns an equivalent
    Callable where that single parameter now has a default value of MATCHED_SYSTEM.
    """
    def _inner(cls, v=MATCHED_SYSTEM):
        # Make sure that v is a KnownSystem
        if v is None:
            return False
        elif type(v) is SystemConfiguration:
            v = match_known_system(v)
        elif type(v) is not KnownSystem:
            raise TypeError(f"Cannot classify system of object type {type(v)}")
        return f(cls, v)
    return _inner


class SystemClassifications:
    """Defines classmethods with the signature Callable[KnownSystem, bool] that returns True or False, representing
    whether or not a KnownSystem satisfies a certain condition.

    This is the equivalent of code.common.constants.AdHocSystemClassification for the old System description from MLPerf
    Inference v1.1."""

    @classmethod
    @_default_on_matched
    def is_hopper(cls, sys):
        return sys.value.get_compute_sm() in [90]

    @classmethod
    @_default_on_matched
    def is_ampere(cls, sys):
        return sys.value.get_compute_sm() in [80, 86, 87]

    @classmethod
    @_default_on_matched
    def is_turing(cls, sys):
        return sys.value.get_compute_sm() == 75

    @classmethod
    @_default_on_matched
    def is_orin(cls, sys):
        return sys in [KnownSystem.Orin]

    @classmethod
    @_default_on_matched
    def is_soc(cls, sys):
        return SystemClassifications.is_orin(sys)

    @classmethod
    @_default_on_matched
    def start_from_device_enabled(cls, sys):
        return sys in [
            KnownSystem.A100_SXM_80GBx1,
            KnownSystem.A100_SXM_80GBx8,
            KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb,
            KnownSystem.A100_SXM4_40GBx1,
            KnownSystem.A100_SXM4_40GBx8,
            KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb,
            KnownSystem.A100_SXM_80GB_ARMx1,
            KnownSystem.A100_SXM_80GB_ARMx8,
        ]

    @classmethod
    @_default_on_matched
    def end_on_device_enabled(cls, sys):
        return sys in [
            KnownSystem.A100_SXM_80GBx1,
            KnownSystem.A100_SXM_80GBx8,
            KnownSystem.A100_SXM_80GB_MIG_1x1g_10gb,
            KnownSystem.A100_SXM4_40GBx1,
            KnownSystem.A100_SXM4_40GBx8,
            KnownSystem.A100_SXM4_40GB_MIG_1x1g_5gb,
            KnownSystem.A100_SXM_80GB_ARMx1,
            KnownSystem.A100_SXM_80GB_ARMx8,
        ]

    @classmethod
    @_default_on_matched
    def intel_openvino(cls, sys):
        return len(sys.value.accelerator_conf.get_accelerators()) == 0 and \
            sys.value.host_cpu_conf.get_architecture() == CPUArchitecture.x86_64

    @classmethod
    @_default_on_matched
    def inferentia_based(cls, sys):
        return sys.value.accelerator_conf.num_inferentia() > 0

    @classmethod
    @_default_on_matched
    def gpu_based(cls, sys):
        return sys in [
            known
            for known in KnownSystem
            if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 0
        ]

    @classmethod
    @_default_on_matched
    def multi_gpu(cls, sys):
        return sys in [
            known
            for known in KnownSystem
            if known.value.accelerator_conf.num_gpus() + known.value.accelerator_conf.num_migs() > 1
        ]
