# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


custom_systems = dict()


# Do not manually edit any lines below this. All such lines are generated via scripts/add_custom_system.py

###############################
### START OF CUSTOM SYSTEMS ###
###############################
custom_systems['SINIAN_VODLA_EFLO_A30x1_MIG'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.490944, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 1}), numa_conf=None, system_id="SINIAN_VODLA_EFLO_A30x1_MIG")
custom_systems['SINIAN_VODLA_EFLO_A30x16'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.490944, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 16}), numa_conf=None, system_id="SINIAN_VODLA_EFLO_A30x16")
custom_systems['SINIAN_VODLA_EFLO_A100x16'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.490944, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_SXM_80GB.value: 16}), numa_conf=None, system_id="SINIAN_VODLA_EFLO_A100x16")
custom_systems['SINIAN_VODLA_EFLO_A100x32'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.490944, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_SXM_80GB.value: 32}), numa_conf=None, system_id="SINIAN_VODLA_EFLO_A100x32")
custom_systems['SINIAN_VODLA_EFLO_A100x8_A30x8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.490944, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_SXM_80GB.value: 8, KnownGPU.A30.value: 8}), numa_conf=None, system_id="SINIAN_VODLA_EFLO_A100x8_A30x8")

###############################
#### END OF CUSTOM SYSTEMS ####
###############################
