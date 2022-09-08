import onnx
import numpy as np
import os
import cv2

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    
    img = np.asarray(img, dtype='float32')
    # normalize image
    means = np.array([124.0, 117.0, 104.0], dtype=np.float32)
    img -= means
    
    if need_transpose:
        img = img.transpose([2, 0, 1])

    return img

def calib_image_preprocess():
    image_size = [224, 224, 3]
    path_dir = "./calib_img"
    file_list = os.listdir(path_dir)

    img_arr=[]

    for img_path in file_list:
        img_path = os.path.join(path_dir, img_path)
        
        img = cv2.imread(img_path)
        processed = pre_process_vgg(img, dims=image_size, need_transpose=True)
        img_arr.append(processed.flatten())

    network_weights = np.concatenate(img_arr, axis = 0)
    return network_weights.tolist()

