#!/usr/bin/env python3
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

import argparse
import ctypes
import os
import re

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
from code.plugin import load_trt_plugin
load_trt_plugin("retinanet")

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt

from code.common import logging, dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.utils import get_dyn_ranges
from code.common.systems.system_list import SystemClassifications
from importlib import import_module
# TODO: Add module needed for retinanet
RetinanetEntropyCalibrator = import_module("code.retinanet.tensorrt.calibrator").RetinanetEntropyCalibrator

INPUT_SHAPE = (3, 800, 800)


class FirstLayerConvActPoolTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        if "Conv_0 + 1783 + Mul_1 + 1785 + Add_2 + Relu_3 + MaxPool_4" in ctx.name:  # Apply to the first layer
            # MLPINF-1833: Disabled CaskConvActPool for TRT 8.5.0.4
            # TRT 8.5.0.4 has a bug with CaskConvActPool which has been fixed since 8.5.0.5
            forbidden_set = {
                -3689373275198309793,  # 0xccccb68da7fc3a5f
                -4219016963003938541,  # 0xc5730a6ceacd8913
                -4709698786673109216,  # 0xbea3c9e81542d720
                8863348452769974412,  # 0x7b00f0752fdcc88c
                -216502845640484144,  # 0xfcfed3cf18bcdad0
                -2840175123683203852,  # 0xd895abc5dcf624f4
                4391967500208500226,  # 0x3cf3672bfafcee02
                -3076721233724812250,  # 0xd54d4a56ceee5426
                8268411641074121664,  # 0x72bf4c9462ed7bc0
                3484514246525022387,  # 0x305b7b3ed6e970b3
                679919370278938099,  # 0x096f8f109d6225f3
                1531503914513228020,  # 0x1540feb22cae60f4
                8162590574723450606,  # 0x714758e16557c6ee
                6137316588591593674,  # 0x552c20eba11d38ca
                -5252194382421728148,  # 0xb71c75095873646c
                -2136593403804660582,  # 0xe2594b9e90c7cc9a
                58603908831090367,  # 0x00d033f1d05396bf
                1454666201826561687,  # 0x1430033412a38e97
                -7506077189063215810,  # 0xd43db7d0f0e3ba45
                -3153162056066942395,  # 0x9521940f435d0c18
                -7700711094551245800,  # 0xf126325c0aa4aa02
                -1070112490556970494,  # 0x97d50e90c139753e
            }
            filtered_idxs = [idx for idx, choice in enumerate(choices) if choice.algorithm_variant.tactic not in forbidden_set]
            to_ret = filtered_idxs
        else:
            # By default, say that all tactics are acceptable:
            to_ret = [idx for idx, _ in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):
        pass


class Retinanet(TensorRTEngineBuilder):
    def __init__(self, args):
        # Retinanet need a bigger workspace
        workspace_size = dict_get(args, "workspace_size", default=(8 << 30))
        logging.info(f"Using workspace size: {workspace_size}")
        super().__init__(args, Benchmark.Retinanet, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/retinanet-resnext50-32x4d/submission/retinanet_resnext50_32x4d_efficientNMS.800x800.onnx")
        self.cache_file = None

        if self.precision == "int8":
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=10)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=50)
            cache_file = dict_get(self.args, "cache_file", default="code/retinanet/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/open-images-v6-mlperf/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "open-images-v6-mlperf/calibration/Retinanet/fp32")

            self.calibrator = RetinanetEntropyCalibrator(calib_image_dir, cache_file, calib_batch_size,
                                                         calib_max_batches, force_calibration, calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file

            # Apply tactic selector bypassing conv act pool for Orin:
            if SystemClassifications.is_orin():
                tactic_selector = FirstLayerConvActPoolTacticSelector()
                self.builder_config.algorithm_selector = tactic_selector

    def initialize(self):
        # Create network.
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_creation_flag)

        parser = trt.OnnxParser(self.network, self.logger)
        success = parser.parse_from_file(self.model_path)

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            dynamic_range_dict = dict()
            if os.path.exists(self.cache_file):
                dynamic_range_dict = get_dyn_ranges(self.cache_file)
                input_dr = dynamic_range_dict.get("images", -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find 'images' in the calibration cache. Exiting...")
                input_tensor.set_dynamic_range(-input_dr, input_dr)
            else:
                print("WARNING: Calibration cache file not found! Calibration is required")
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        self.initialized = True
