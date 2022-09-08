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

from operator import index
import onnx
import onnxruntime
import netron
import torch
import torch.nn as nn
import numpy as np


import copy
def delete_node(model, nodename):
  for i in range(20):
    for node in model.graph.node:
        if node.name == nodename:
          model.graph.node.remove(node)

def add_constant(model, name, value):
    value1 = onnx.helper.make_tensor(name = name, data_type=onnx.TensorProto.INT64, dims=value.shape, vals=value)
    const1 = onnx.helper.make_node(op_type ='Constant', inputs=[], outputs=[name], value=value1)
    model.graph.node.append(const1)


model = onnx.load("bert_mlpf_quant2.onnx")

node_to_remove = []
for i in range(22):
  node_to_remove.append(str(i))

for nodename in node_to_remove:
  delete_node(model, nodename)


# 增加input4
input4 = copy.deepcopy(model.graph.input[0]) # 21
input4.name = "position_ids"
model.graph.input.append(input4)
# 接上input4

for node in model.graph.node:
  if node.name == "23":
    node.input[1] = "position_ids"

for ii in model.graph.input:
  ii.type.tensor_type.shape.dim[0].dim_param = '1'
  ii.type.tensor_type.shape.dim[1].dim_param = '512'
  ii.type.tensor_type.elem_type = onnx.TensorProto.INT32

# 修改mask fill input shape
model.graph.input[1].type.tensor_type.shape.dim[0].dim_param = '1'
model.graph.input[1].type.tensor_type.shape.dim[1].dim_param = '6'


input2 = model.graph.input.pop(1)
input4 = model.graph.input.pop()
model.graph.input.insert(1, input4)
model.graph.input.insert(3, input2)

position_ids = model.graph.input.pop(1)
segment_ids = model.graph.input.pop(1)

model.graph.input.insert(1, segment_ids)
model.graph.input.insert(2, position_ids)

# 增加算子
starts = np.array([0], dtype=np.int64)
add_constant(model, 'starts', starts)
ends = np.array([1], dtype=np.int64)
add_constant(model, 'ends', ends)
axes = np.array([1], dtype=np.int64)
add_constant(model, 'axes', axes)
node1 = onnx.helper.make_node('Slice', inputs=['input_mask', 'starts', 'ends', 'axes'], outputs=['A'])
model.graph.node.append(node1)

new_shape1 = np.array([256, 32], dtype=np.int64)
add_constant(model, 'new_shape1', new_shape1)
node2 = onnx.helper.make_node('Expand', inputs=['A', 'new_shape1'], outputs=['B'])
model.graph.node.append(node2)

node3 = onnx.helper.make_node('Cast', inputs=['B'], outputs=['C'], to=onnx.TensorProto.FLOAT)
model.graph.node.append(node3)

####################################################################
starts = np.array([0, 0], dtype=np.int64)
add_constant(model, 'starts1', starts)
ends = np.array([1, 1], dtype=np.int64)
add_constant(model, 'ends1', ends)
axes = np.array([0, 1], dtype=np.int64)
add_constant(model, 'axes1', axes)
new_shape2 = np.array([1, 1, 1, 1], dtype=np.int64)
add_constant(model, 'new_shape2', new_shape2)

add_list = []
for i in range(24):
  node4 = onnx.helper.make_node('Slice', inputs=['C', 'starts1', 'ends1', 'axes1'], outputs=['D' + str(i)])
  model.graph.node.append(node4)

  node5 = onnx.helper.make_node('Expand', inputs=['D' + str(i), 'new_shape2'], outputs=['E' + str(i)])
  model.graph.node.append(node5)
  add_node_index = 95 + 114*i
  add_list.append(str(add_node_index))


for node in model.graph.node:
    if node.name in add_list:
      idex = add_list.index(node.name)
      node.input[1] = 'E'+str(idex)

flat_node = onnx.helper.make_node(
      "BRFlat",
      inputs=['3172'],
      outputs=['3175'],
  )

model.graph.node.insert(2817, flat_node)

for node in model.graph.node:
  if node.name == "2777":
    node.input[0] = '3175'

path_new = "int8_4_inputs.onnx"
print(model.graph.input[1].name)
print(model.graph.input[1].type.tensor_type.shape.dim[0].dim_param)
print(model.graph.input[1].type.tensor_type.shape.dim[1].dim_param)

onnx.save(model, path_new)

