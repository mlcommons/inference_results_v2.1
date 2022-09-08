#
# Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
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
# 

import onnx, argparse
import numpy as np

def parse_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--input_model", help="model path")
    args_parser.add_argument("--output_model", help="model path")
    args = args_parser.parse_args()
    return args

def post_process(input_model, output_model):
  model = onnx.load(input_model)

  model.graph.input[0].type.tensor_type.elem_type = 3
  model.graph.input[0].name = "input_tensor:0_quantized"

  # print(model.graph.input[0])

  for node in model.graph.node:
    if node.name == "input_tensor:0_QuantizeLinear":
      model.graph.node.remove(node)

  initializes_to_delete = ["input_tensor:0_scale", "input_tensor:0_zero_point"]

  model = onnx.shape_inference.infer_shapes(model)

  graph = model.graph
  nodes = graph.node
  fake_node_name = [
    'Conv_nc_rename_8_quant',
    'Conv_nc_rename_14_quant',
    'Conv_nc_rename_26_quant',
    'Conv_nc_rename_32_quant',
    'Conv_nc_rename_38_quant',
    'Conv_nc_rename_50_quant',
    'Conv_nc_rename_56_quant',
    'Conv_nc_rename_62_quant',
    'Conv_nc_rename_68_quant',
    'Conv_nc_rename_74_quant',
    'Conv_nc_rename_86_quant',
    'Conv_nc_rename_92_quant'
  ]

  for node in nodes:
      if node.name in fake_node_name:
          del node.input[-1]
          # print(node.input)

  onnx.save(model, output_model)

if __name__ == "__main__":

  args = parse_args()
  model_path = args.input_model
  output_model = "models/resnet50_int8_delete_quantize_linear_v2_0803.onnx"
  output_model = args.output_model

  post_process(model_path, output_model)

  print("success \ngenerate quantized model")
