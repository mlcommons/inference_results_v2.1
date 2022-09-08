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

from pickletools import uint8
import cv2
import numpy as np
import os

path = r"/home/hsun/workspace/ramfs/suinfer_preprocess/data/golden"

input_file = os.path.join(path, "outputs_origin/ILSVRC2012_val_00000001.JPEG")
output_file = os.path.join(path, "outputs_convert/ILSVRC2012_val_00000001.JPEG")

print(input_file)
npy_input = np.fromfile(input_file, dtype='uint8')
print(npy_input.shape)
npy_input = npy_input.reshape(3, 224, 224)
print(npy_input.shape)
# npy_input = np.pad(npy_input, ((0,0), (3,2), (3,2)), 'constant', constant_values=0)
# print(npy_input.shape)
npy_input = npy_input.transpose([1, 2, 0])
print(npy_input.shape)
npy_input = np.pad(npy_input, ((3,2), (3,2), (0,0)), 'constant', constant_values=0)
print(npy_input.shape)

npy_input.astype('uint8').tofile(output_file)


