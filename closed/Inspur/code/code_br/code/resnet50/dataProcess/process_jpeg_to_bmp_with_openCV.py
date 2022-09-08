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

import os
import cv2

ISLVRC_path = '/home/br104/suinfer/njh/py_workspace/ISLVRC'
saveISLVRC_path = '/home/br104/suinfer/njh/py_workspace/ISLVRC_bmp'
list_dir = os.listdir(ISLVRC_path)

for index, img in enumerate(list_dir):
    img_path = os.path.join(ISLVRC_path , img)
    save_path = os.path.join(saveISLVRC_path, img.split('.')[0]+'.bmp')
    print(index)

    img_data = cv2.imread(img_path)
    cv2.imwrite(save_path, img_data)

