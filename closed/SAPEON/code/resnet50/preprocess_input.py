import sys
import cv2
import os
import numpy as np
import struct

import argparse
from tqdm import tqdm
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='input imagenet dir path')
parser.add_argument('-o', '--output', required=True, help='image preprocessed result')
args = parser.parse_args()

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

def preprocess_vgg(in_out_paths, dims=[224, 224, 3], need_transpose=False):
    input_path = in_out_paths[0]
    out_path = in_out_paths[1]
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width, _ = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    
    img = np.asarray(img, dtype='float32')
    # normalize image
    means = np.array([124.0, 117.0, 104.0], dtype=np.float32)
    img -= means
    
    iscale = 0.840726
    
    img = img * iscale
    img = np.clip(np.round(img), -127, 127)
    img = np.asarray(img, dtype=np.int8)
        
    if need_transpose:
        img = img.transpose([1, 0, 2])
    img = img.flatten()

    s = struct.pack('b' * len(img), *img)
    f = open(out_path, 'wb')
    f.write(s)
    f.close()
    return img

def main():
    input_dir = args.input
    output_dir = args.output

    if os.path.exists(output_dir):
        print(output_dir, 'already exists! change -o argument!')
        sys.exit()
    os.makedirs(output_dir)

    input_filenames = os.listdir(input_dir)
    input_paths = [os.path.join(input_dir, filename) for filename in input_filenames]
    output_paths = [os.path.join(output_dir, filename.replace('JPEG', 'bin', 1)) for filename in input_filenames]
    pair_paths = list(zip(input_paths, output_paths))

    with mp.Pool(os.cpu_count()) as p:
        r = list(tqdm(p.imap(preprocess_vgg, pair_paths), total=len(pair_paths)))
    print('preprocessing COMPLETE')

if __name__ == '__main__':
    main()