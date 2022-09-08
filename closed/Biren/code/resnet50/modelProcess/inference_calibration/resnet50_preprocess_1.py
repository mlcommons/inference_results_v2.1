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

import onnx, sys
# import netron
import numpy as np

def weight_gen(weight_name, weight_shape):
  weight_data = np.zeros(weight_shape).astype(np.float32)
  channel = weight_shape[1]
  kernel_index = 0
  for i in range(channel):
    weight_data[i][kernel_index][0][0] = 1
    kernel_index +=1

  assert(weight_data.sum() == channel)
  weight_tensor=onnx.helper.make_tensor(weight_name, onnx.TensorProto.FLOAT, weight_shape, weight_data.tobytes(), raw = True)
  return weight_tensor

def bias_gen(bias_name, bias_shape):
  bias_data = np.zeros(bias_shape).astype(np.float32)

  bias_tensor = onnx.helper.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias_shape, bias_data.tobytes(), raw = True)
  return bias_tensor

def conv_gen(input_name, weight_name, bias_name, output_name):
  conv_node = onnx.helper.make_node(
              'Conv',
              inputs=[input_name, weight_name, bias_name],
              outputs=[output_name],
              kernel_shape=[1, 1],
              pads = [0,0,0,0]
            )

  return conv_node

def onnx_insert_conv(model, input_tensor_name, weight_shape):
  output_tensor_name = "equal_" + input_tensor_name
  weight_name = "weight_" + input_tensor_name
  bias_name = "bias_" + input_tensor_name

  weight = weight_gen(weight_name, weight_shape)

  bias_shape = [weight_shape[0]]
  bias = bias_gen(bias_name, bias_shape)

  model.graph.initializer.append(weight)
  model.graph.initializer.append(bias)

  conv = conv_gen(input_tensor_name, weight_name, bias_name, output_tensor_name)

  label = False

  for node in model.graph.node:
    for input in node.input:
      if input == input_tensor_name and node.op_type != "Conv":
        node.input.remove(input)
        node.input.append(output_tensor_name)
        label = True

  assert(label)

  model.graph.node.append(conv)
  return model


def set_input_shape(model, input_name, input_shape):
  for input in model.graph.input:
    model.graph.input.remove(input)

  input_layer_value_info= onnx.helper.make_tensor_value_info(input_name, 1, input_shape)
  model.graph.input.append(input_layer_value_info)

  return model

def shape_inference(src_path, dst_path):
  model = onnx.load(src_path)
  model = set_input_shape(model, "input_tensor:0", [1, 3, 224, 224])
  model = onnx.shape_inference.infer_shapes(model)
  onnx.save(model, dst_path)
  # netron.start(dst_path)


if __name__ == "__main__":
  # path = "/home/mtn/suinfer_temp/resnet50.onnx"
  # new_path = "resnet50_insert_conv.onnx"
  resnet50_mlpert_onnx = "resnet50_v1.onnx"
  resnet50_mlperf_onnx_equal_conv = "resnet50_mlperf_equal_conv.onnx"
  
  resnet50_mlpert_onnx = sys.argv[1]
  resnet50_mlperf_onnx_equal_conv = sys.argv[2]

  # shape_inference(resnet50_mlpert_onnx, resnet50_mlpert_onnx)
  model = onnx.load(resnet50_mlpert_onnx)

  # use for resnet50.onnx
  # conv_info = [("336", [256, 256, 1, 1]),
  #              ("346", [256, 256, 1, 1]),
  #              ("368", [512, 512, 1, 1]),
  #              ("378", [512, 512, 1, 1]),
  #              ("388", [512, 512, 1, 1]),
  #              ("410", [1024, 1024, 1, 1]),
  #              ("420", [1024, 1024, 1, 1]),
  #              ("430", [1024, 1024, 1, 1]),
  #              ("440", [1024, 1024, 1, 1]),
  #              ("450", [1024, 1024, 1, 1]),
  #              ("472", [2048, 2048, 1, 1]),
  #              ("482", [2048, 2048, 1, 1]),
  #              ]

  conv_info = [("resnet_model/Relu_3:0", [256, 256, 1, 1]),
               ("resnet_model/Relu_6:0", [256, 256, 1, 1]),
               ("resnet_model/Relu_12:0", [512, 512, 1, 1]),
               ("resnet_model/Relu_15:0", [512, 512, 1, 1]),
               ("resnet_model/Relu_18:0", [512, 512, 1, 1]),
               ("resnet_model/Relu_24:0", [1024, 1024, 1, 1]),
               ("resnet_model/Relu_27:0", [1024, 1024, 1, 1]),
               ("resnet_model/Relu_30:0", [1024, 1024, 1, 1]),
               ("resnet_model/Relu_33:0", [1024, 1024, 1, 1]),
               ("resnet_model/Relu_36:0", [1024, 1024, 1, 1]),
               ("resnet_model/Relu_42:0", [2048, 2048, 1, 1]),
               ("resnet_model/Relu_45:0", [2048, 2048, 1, 1]),
               ]

  for item in conv_info:
    model = onnx_insert_conv(model, item[0], item[1])

  model = onnx.shape_inference.infer_shapes(model)

  onnx.save(model, resnet50_mlperf_onnx_equal_conv)
  # netron.start(resnet50_mlperf_onnx_equal_conv)

