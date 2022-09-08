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

custom_systems['A100_PCIE_80GBx20'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6346 CPU @ 3.10GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.113462056, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 20}), numa_conf=None, system_id="A100_PCIE_80GBx20")
custom_systems['A100_SXM4_80GBX1_CUSTOM'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8368Q CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=20, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.113434772, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM4-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=500.0, pci_id="0x20B210DE", compute_sm=80): 1}), numa_conf=None, system_id="A100_SXM4_80GBX1_CUSTOM")
custom_systems['A100_SXM4_80GBx8_Custom'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8368Q CPU @ 2.60GHz", architecture=CPUArchitecture.x86_64, core_count=20, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.1134547080000003, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM4-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=500.0, pci_id="0x20B210DE", compute_sm=80): 8}), numa_conf=None, system_id="A100_SXM4_80GBx8_Custom")

###############################
#### END OF CUSTOM SYSTEMS ####
###############################
