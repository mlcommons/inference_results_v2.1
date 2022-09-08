# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

import ctypes
import os

from code.plugin.plugin_map import G_PLUGIN_MAP


def load_trt_plugin(network_name: str) -> None:
    if network_name not in G_PLUGIN_MAP:
        raise KeyError(f"Requested {network_name} does not have any plugin. Valid options: {list(G_PLUGIN_MAP.keys())}")
    if G_PLUGIN_MAP[network_name] is not None:
        for plugin_name, plugin_path in G_PLUGIN_MAP[network_name].items():
            plugin_path = "build/plugins/" + plugin_path
            if not os.path.isfile(plugin_path):
                raise IOError(f"Failed to load from ({plugin_path}).\nPlease build the {plugin_name} plugin for {network_name}.\n")
            ctypes.CDLL(plugin_path)
            print(f"Loaded {plugin_name} plugin from {plugin_path} for {network_name}")
