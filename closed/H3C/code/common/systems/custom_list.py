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

custom_systems['R5500G5_A100_SXM_80GB_CTSx1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8378A CPU @ 3.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056499132, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_SXM_80GB_500.value: 1}), numa_conf=None, system_id="R5500G5_A100_SXM_80GB_CTSx1")

custom_systems['A2x5_R4900G5'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz", architecture=CPUArchitecture.x86_64, core_count=40, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.11343948, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A2.value: 5}), numa_conf=None, system_id="A2x5_R4900G5")

custom_systems['A30x3_R4900G5'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz", architecture=CPUArchitecture.x86_64, core_count=40, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564758840000001, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 3}), numa_conf=None, system_id="A30x3_R4900G5")

custom_systems['A100_PCIe_80GBX20'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz", architecture=CPUArchitecture.x86_64, core_count=40, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.1134471519999996, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 20}), numa_conf=None, system_id="A100_PCIe_80GBX20")

custom_systems['R5300G5_A30x8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.05649934, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 8}), numa_conf=None, system_id="R5300G5_A30x8")

custom_systems['R5300G5_A100_PCIE_80GBX4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=528.022652, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 4}), numa_conf=None, system_id="R5300G5_A100_PCIE_80GBX4")

custom_systems['A30x1_R4950G5'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7763 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=64, threads_per_core=1): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=131.80338, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 1}), numa_conf=None, system_id="A30x1_R4950G5")

custom_systems['A100_PCIe_80GBx1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7763 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=64, threads_per_core=1): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=131.80338, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 1}), numa_conf=None, system_id="A100_PCIe_80GBx1")

custom_systems['A2x6_R4950G5'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7513 32-Core Processor", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056640284, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A2.value: 6}), numa_conf=None, system_id="A2x6_R4950G5")

custom_systems['A30x2_R4950G5'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7513 32-Core Processor", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056640284, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A30.value: 2}), numa_conf=None, system_id="A30x2_R4950G5")

custom_systems['A2x2'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7413 24-Core Processor", architecture=CPUArchitecture.x86_64, core_count=24, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056566852, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A2.value: 2}), numa_conf=None, system_id="A2x2")

custom_systems['R5300G5_A100_SXM_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8362 CPU @ 2.80GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564835600000002, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_SXM_80GB.value: 4}), numa_conf=None, system_id="R5300G5_A100_SXM_80GBx4")

###############################
#### END OF CUSTOM SYSTEMS ####
###############################
