#! /usr/bin/env python3
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

__doc__ = """
Perform naive inference for 3D-UNet KiTS19 benchmark, using TensorRT engine
Useful for quick studies on performance/accuracy.

Example command:
    python -m code.3d-unet.tensorrt.infer --verbose --engine_file </path/to/engine/file>

    # Use full-image inference instead of sliding window
    python -m code.3d-unet.tensorrt.infer --verbose\
        --engine_file build/engines/A100-SXM-80GBx1/3d-unet/Offline/3d-unet-Offline-gpu-b1-int8.lwis_k_99_MaxP.plan\
        --batch_size=1 --num_samples=3\
        --full_image

    # Use preconditioned Gaussian patches, instead of generating them on-the-fly
    python -m code.3d-unet.tensorrt.infer --verbose\
        --engine_file build/engines/A100-SXM-80GBx1/3d-unet/Offline/3d-unet-Offline-gpu-b1-int8.lwis_k_99_MaxP.plan\
        --preconditioned

    # Use INT8 LINEAR input, slice it for sliding window, and zero-pad to NC/32DHW32 on-the-fly
    python -m code.3d-unet.tensorrt.infer --verbose\
        --engine_file build/engines/A100-SXM-80GBx1/3d-unet/Offline/3d-unet-Offline-gpu-b1-int8.lwis_k_99_MaxP.plan\
        --zero_pad_int8_linear_input

Also infer.py can use custom CUDA kernels:
    # Build the CUBin for custom CUDA kernels
    nvcc --cubin -arch sm_80 -I/usr/local/lib/python3.8/dist-packages/pycuda/cuda code/3d-unet/tensorrt/unet3d_sw.cu\
        -o code/3d-unet/tensorrt/unet3d_sw.cubin

    # Use custom CUDA kernels in infer.py
    python -m code.3d-unet.tensorrt.infer --verbose\
        --engine_file build/engines/A100-SXM-80GBx1/3d-unet/Offline/3d-unet-Offline-gpu-b1-int8.lwis_k_99_MaxP.plan\
        --preconditioned\
        --use_cuda_kernels
"""


# Load .so plugin files first
from code.plugin import load_trt_plugin
load_trt_plugin("3d-unet-kits")

import ctypes
import os

import argparse
import pickle
import shutil
import time
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple, Union

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import nibabel as nib
    import pandas as pd
    from scipy import signal
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule


import code.common.arguments as common_args
from code.common.runner import EngineRunner, get_input_format
from code.common import logging


# KiTS19 constants
ROI_SHAPE = [128, 128, 128]
SLIDE_OVERLAP_FACTOR = 0.5

# $(DTYPE) mapping to numpy dtype
dtype_map = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64
}


def gaussian_kernel(n: int, std: float) -> np.ndarray:
    """
    Returns gaussian kernel; std is standard deviation and n is number of points
    """
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return gaussian3D


def apply_norm_map(image: np.ndarray, norm_map: np.ndarray) -> np.ndarray:
    """
    Applies normal map norm_map to image and return the outcome
    """
    image /= norm_map
    return image


def apply_argmax(image: np.ndarray) -> np.ndarray:
    """
    Returns indices of the maximum values along the channel axis
    Input shape is (bs=1, channel=3, (image_shape)), float -- sub-volume inference result
    Output shape is (bs=1, channel=1, (image_shape)), integer -- segmentation result
    """
    channel_axis = 1
    image = np.argmax(image, axis=channel_axis).astype(np.uint8)
    image = np.expand_dims(image, axis=0)

    return image


def apply_argmax_cuda(image: np.ndarray) -> np.ndarray:
    """
    Returns indices of the maximum values along the channel axis
    Input shape is (bs=1, channel=3, (image_shape)), float -- sub-volume inference result
    Output shape is (bs=1, channel=1, (image_shape)), integer -- segmentation result
    This is CUDA version
    """
    cumod = pycuda.driver.module_from_file('code/3d-unet/tensorrt/unet3d_sw.cubin')

    image_shape = image.shape[2:]
    block = (image_shape[2], 1, 1)
    grid = (image_shape[0], image_shape[1], 1)

    h_param = np.array(
        list(image_shape) + [np.prod(image_shape)] +
        [0, 0, 0] +
        [ROI_SHAPE[0], np.prod(ROI_SHAPE)] +
        [1, 3], dtype=np.int32)

    h_res = np.empty([1, 1] + list(image_shape), dtype=np.int8)

    d_in = cuda.mem_alloc(image.nbytes)
    d_out = cuda.mem_alloc(h_res.nbytes)
    d_param = cuda.mem_alloc(h_param.nbytes)

    cuda.memcpy_htod(d_in, image)
    cuda.memcpy_htod(d_param, h_param)

    func = cumod.get_function('UNet3DKiTS19ArgMaxKernel')
    func(d_in, d_out, d_param, grid=grid, block=block)

    cuda.memcpy_dtoh(h_res, d_out)

    return h_res


def finalize(image: np.ndarray,
             norm_map: np.ndarray,
             normalize: bool = True,
             use_cuda_kernels: bool = False) -> np.ndarray:
    """
    Finalizes results obtained from sliding window inference
    """
    # NOTE: layout is assumed to be linear (NCDHW) always
    # apply norm_map
    if normalize:
        image = apply_norm_map(image, norm_map)

    # argmax
    if use_cuda_kernels:
        image = apply_argmax_cuda(image)
    else:
        image = apply_argmax(image)

    return image


def get_input_slice_cuda(image: np.ndarray, offsets: List) -> np.ndarray:
    """
    slice the LINEAR input image and return the CDHW32 slice using CUDA kernel
    """
    cumod = pycuda.driver.module_from_file('code/3d-unet/tensorrt/unet3d_sw.cubin')

    assert image.shape[1] == 1, 'LINEAR input is expected here'
    image_shape = image.shape[2:]
    block = (ROI_SHAPE[2], 1, 1)
    grid = (ROI_SHAPE[0], ROI_SHAPE[1], 1)

    h_param = np.array(
        list(image_shape) + [np.prod(image_shape)] +
        offsets +
        [ROI_SHAPE[0], np.prod(ROI_SHAPE)] +
        [1, 3], dtype=np.int32)

    h_res = np.zeros([1] + list(ROI_SHAPE) + [32], dtype=np.int8)

    d_in = cuda.mem_alloc(image.nbytes)
    d_out = cuda.mem_alloc(h_res.nbytes)
    d_param = cuda.mem_alloc(h_param.nbytes)

    cuda.memcpy_htod(d_in, image)
    cuda.memcpy_htod(d_param, h_param)
    cuda.memcpy_htod(d_out, h_res)  # instead of memset, h_res zero'ed out

    func = cumod.get_function('UNet3DKiTS19SliceKernelI8CDHW32')
    func(d_in, d_out, d_param, grid=grid, block=block)

    cuda.memcpy_dtoh(h_res, d_out)

    return h_res


def apply_patch_cuda(result: np.ndarray,
                     slice: np.ndarray,
                     patch: np.ndarray,
                     offsets: List) -> np.ndarray:
    """
    Apply preconditioned Gaussian patch to sliding window result and accumulate final result
    """
    cumod = pycuda.driver.module_from_file('code/3d-unet/tensorrt/unet3d_sw.cubin')

    image_shape = result.shape[2:]
    block = (ROI_SHAPE[2], 1, 1)
    grid = (ROI_SHAPE[0], ROI_SHAPE[1], 1)

    h_param = np.array(
        list(image_shape) + [np.prod(image_shape)] +
        offsets +
        [ROI_SHAPE[0], np.prod(ROI_SHAPE)] +
        [1, 3], dtype=np.int32)

    d_in = cuda.mem_alloc(slice.nbytes)
    d_patch = cuda.mem_alloc(patch.nbytes)
    d_out = cuda.mem_alloc(result.nbytes)
    d_param = cuda.mem_alloc(h_param.nbytes)

    cuda.memcpy_htod(d_in, slice)
    cuda.memcpy_htod(d_patch, patch)
    cuda.memcpy_htod(d_out, result)
    cuda.memcpy_htod(d_param, h_param)

    func = cumod.get_function('UNet3DKiTS19PatchKernel')
    func(d_in, d_patch, d_out, d_param, grid=grid, block=block)

    cuda.memcpy_dtoh(result, d_out)

    return result


def prepare_arrays(image: np.ndarray,
                   roi_shape: List = ROI_SHAPE,
                   tgt_dtype: np.dtype = np.float16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns empty arrays required for sliding window inference such as:
    - result array where sub-volume inference results are gathered
    - norm_map where normal map is constructed upon
    - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"

    image_shape = list(image.shape[2:])

    result = np.zeros(shape=(1, 3, *image_shape), dtype=tgt_dtype)
    norm_map = np.zeros_like(result)
    norm_patch = gaussian_kernel(
        roi_shape[0], 0.125 * roi_shape[0]).astype(tgt_dtype)

    return result, norm_map, norm_patch


def get_slice_for_sliding_window(image: np.ndarray,
                                 roi_shape: List = ROI_SHAPE,
                                 overlap: int = SLIDE_OVERLAP_FACTOR) -> Tuple[int, int, int, int]:
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = list(image.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) //
            strides[i] + 1 for i in range(dim)]
    i_range = range(0, strides[0] * size[0], strides[0])
    j_range = range(0, strides[1] * size[1], strides[1])
    k_range = range(0, strides[2] * size[2], strides[2])
    total_itr_left = len(i_range) * len(j_range) * len(k_range)
    for i in i_range:
        for j in j_range:
            for k in k_range:
                total_itr_left -= 1
                yield i, j, k, total_itr_left


def prepare_arrays_preconditioned(image: np.ndarray,
                                  roi_shape: List = ROI_SHAPE,
                                  tgt_dtype: np.dtype = np.float16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns empty arrays required for sliding window inference such as:
    - result array where sub-volume inference results are gathered
    - norm_map where normal map is constructed upon
    - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"

    image_shape = list(image.shape[2:])

    result = np.zeros(shape=(1, 3, *image_shape), dtype=tgt_dtype)
    norm_patches = list()

    preproc_dir_env = os.getenv("PREPROCESSED_DATA_DIR", default="build/preprocessed_data")
    patch_npy_path = Path(preproc_dir_env, "KiTS19", "etc", "gaussian_patches.npy")

    if patch_npy_path.is_file():
        norm_patches = np.load(patch_npy_path)
    else:
        norm_map = np.zeros_like(result)
        norm_patch = gaussian_kernel(
            roi_shape[0], 0.125 * roi_shape[0]).astype(tgt_dtype)

        norm_patches = [None, ] * 27

        for i, j, k, l in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            norm_map_slice = norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_map_slice += norm_patch

        # each dim is: 0 -- start corner, 1 -- middle, 2 -- end corner
        for i, j, k, l in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            my_id = [0 if i == 0 else 2 if i + ROI_SHAPE[0] == image_shape[0] else 1,
                     0 if j == 0 else 2 if j + ROI_SHAPE[1] == image_shape[1] else 1,
                     0 if k == 0 else 2 if k + ROI_SHAPE[2] == image_shape[2] else 1]
            patch_id = my_id[0] * 9 + my_id[1] * 3 + my_id[2]
            my_slice = norm_map[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)]

            norm_patches[patch_id] = norm_patch / my_slice

    return result, norm_patches


def get_slice_for_sliding_window_preconditioned(image: np.ndarray,
                                                roi_shape: List = ROI_SHAPE,
                                                overlap: int = SLIDE_OVERLAP_FACTOR) -> Tuple[int, int, int, int, int]:
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = list(image.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) //
            strides[i] + 1 for i in range(dim)]
    i_range = range(0, strides[0] * size[0], strides[0])
    j_range = range(0, strides[1] * size[1], strides[1])
    k_range = range(0, strides[2] * size[2], strides[2])
    total_itr_left = len(i_range) * len(j_range) * len(k_range)
    for i in i_range:
        for j in j_range:
            for k in k_range:
                total_itr_left -= 1
                my_id = [0 if i == 0 else 2 if i + ROI_SHAPE[0] == image_shape[0] else 1,
                         0 if j == 0 else 2 if j + ROI_SHAPE[1] == image_shape[1] else 1,
                         0 if k == 0 else 2 if k + ROI_SHAPE[2] == image_shape[2] else 1]
                patch_id = my_id[0] * 9 + my_id[1] * 3 + my_id[2]
                yield i, j, k, patch_id, total_itr_left


def infer_single_query_preconditioned(runner: EngineRunner,
                                      query: np.ndarray,
                                      dtypestr: str,
                                      fmtstr: str,
                                      batch_size: int = 1,
                                      zero_pad: bool = False,
                                      use_cuda_kernels: bool = False) -> Tuple[np.ndarray, int]:
    """
    Performs inference upon data and summarize work with mystr
    Naive implementation of sliding window inference on sub-volume for predetermined
    ROI (Region of Interest) shape is handled here
    This version uses Gaussian kernel patches that are preconditioned and stored in storage
    Parameters
    ----------
        runner:
            Runner running TRT engine
        query: numpy.ndarray
            KiTS19 image to infer
        dtypestr: str
            Input dtype in string
        fmtstr: str
            Input format in string
        batch_size: int
            Batch size to use for sub-volume inference
        zero_pad: bool
            Whether to zero-pad input on-the-fly
        use_cuda_kernels: bool
            Whether to use custom CUDA kernels
    """

    # Inference output format is LINEAR
    # 1ch FP16/INT8 input all appear to be LINEAR
    axes_to_linear = (0, 1, 2, 3, 4)
    axes_from_linear = (0, 1, 2, 3, 4)

    if fmtstr in ["fp16_dhwc8", "int8_cdhw32"] and not zero_pad:
        axes_to_linear = (0, 4, 1, 2, 3)
        axes_from_linear = (0, 2, 3, 4, 1)

    # slide window inference output shape
    sw_out_shape = [1, 3] + ROI_SHAPE
    sw_out_length = np.prod(sw_out_shape)

    # prepare arrays
    image = query[np.newaxis, ...]
    # FIXME: supported format? Maybe only support LINEAR?
    image_linear = image.transpose(axes_to_linear)
    result, norm_patches = prepare_arrays_preconditioned(image_linear, ROI_SHAPE, np.float16)

    # t_* need to be LINEAR format
    # FIXME: maybe I should use GPU memory for t_*?
    t_image = image_linear
    t_result = result

    # sliding window inference
    subvol_cnt = 0
    batch_cnt = 0
    r_slices = list()
    i_slices = list()
    o_slices = list()
    for i, j, k, patch_key, l in get_slice_for_sliding_window_preconditioned(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        t_norm_patch = norm_patches[patch_key]

        subvol_cnt += 1
        batch_cnt += 1
        result_slice = t_result[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        input_slice = None
        if use_cuda_kernels and fmtstr == "int8" and zero_pad:
            input_slice = get_input_slice_cuda(t_image, [i, j, k])
        else:
            input_slice = t_image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)].transpose(axes_from_linear)
            if zero_pad:
                input_slice = np.pad(input_slice, ((0, 0), (0, 31), (0, 0), (0, 0), (0, 0)),
                                     mode="constant").transpose(0, 2, 3, 4, 1)

        r_slices.append(result_slice)
        i_slices.append(input_slice)
        o_slices.append([i, j, k])

        if (batch_cnt >= batch_size or l <= 0):
            runner_input = np.ascontiguousarray(np.concatenate(i_slices, axis=0).astype(dtype_map[dtypestr]))
            runner_output = runner([runner_input], batch_size)
            for l in range(batch_cnt):
                slice_out = np.array(runner_output[0][sw_out_length * l:sw_out_length * (l + 1)]).reshape(sw_out_shape)
                if use_cuda_kernels:
                    t_result = apply_patch_cuda(t_result, slice_out, t_norm_patch, o_slices[l])
                else:
                    r_slices[l] += slice_out * t_norm_patch
            batch_cnt = 0
            r_slices = list()
            i_slices = list()
            o_slices = list()

    result = t_result

    final_result = np.squeeze(finalize(result, None, normalize=False, use_cuda_kernels=use_cuda_kernels), axis=0)
    return final_result, subvol_cnt


def infer_single_query(runner: EngineRunner,
                       query: np.ndarray,
                       dtypestr: str,
                       fmtstr: str,
                       batch_size: int = 1,
                       zero_pad: bool = False,
                       use_cuda_kernels: bool = False) -> Tuple[np.ndarray, int]:
    """
    Performs inference upon data and summarize work with mystr
    Naive implementation of sliding window inference on sub-volume for predetermined
    ROI (Region of Interest) shape is handled here
    This generates Gaussian kernel patches on-the-fly
    Parameters
    ----------
        runner:
            Runner running TRT engine
        query: numpy.ndarray
            KiTS19 image to infer
        dtypestr: str
            Input dtype in string
        fmtstr: str
            Input format in string
        batch_size: int
            Batch size to use for sub-volume inference
        zero_pad: bool
            Whether to zero-pad input on-the-fly
        use_cuda_kernels: bool
            Whether to use custom CUDA kernels
    """

    # Inference output format is LINEAR
    # 1ch FP16/INT8 input all appear to be LINEAR

    axes_to_linear = (0, 1, 2, 3, 4)
    axes_from_linear = (0, 1, 2, 3, 4)

    if fmtstr in ["fp16_dhwc8", "int8_cdhw32"] and not zero_pad:
        axes_to_linear = (0, 4, 1, 2, 3)
        axes_from_linear = (0, 2, 3, 4, 1)

    # slide window inference output shape
    sw_out_shape = [1, 3] + ROI_SHAPE
    sw_out_length = np.prod(sw_out_shape)

    # prepare arrays
    image = query[np.newaxis, ...]
    # FIXME: supported format? Maybe only support LINEAR?
    image_linear = image.transpose(axes_to_linear)
    result, norm_map, norm_patch = prepare_arrays(image_linear, ROI_SHAPE, np.float16)

    # t_* need to be LINEAR format
    # FIXME: maybe I should use GPU memory for t_*?
    t_image = image_linear
    t_result = result
    t_norm_map = norm_map
    t_norm_patch = norm_patch

    # sliding window inference
    subvol_cnt = 0
    batch_cnt = 0
    r_slices = list()
    i_slices = list()
    n_slices = list()
    for i, j, k, l in get_slice_for_sliding_window(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        subvol_cnt += 1
        batch_cnt += 1
        result_slice = t_result[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        input_slice = None
        if use_cuda_kernels and fmtstr == "int8" and zero_pad:
            input_slice = get_input_slice_cuda(t_image, [i, j, k])
        else:
            input_slice = t_image[
                ...,
                i:(ROI_SHAPE[0] + i),
                j:(ROI_SHAPE[1] + j),
                k:(ROI_SHAPE[2] + k)].transpose(axes_from_linear)
            if zero_pad:
                input_slice = np.pad(input_slice, ((0, 0), (0, 31), (0, 0), (0, 0), (0, 0)),
                                     mode="constant").transpose(0, 2, 3, 4, 1)

        norm_map_slice = t_norm_map[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)]

        r_slices.append(result_slice)
        i_slices.append(input_slice)
        n_slices.append(norm_map_slice)

        if (batch_cnt >= batch_size or l <= 0):
            runner_input = np.ascontiguousarray(np.concatenate(i_slices, axis=0).astype(dtype_map[dtypestr]))
            runner_output = runner([runner_input], batch_size)
            for i in range(batch_cnt):
                slice_out = np.array(runner_output[0][sw_out_length * i:sw_out_length * (i + 1)]).reshape(sw_out_shape)
                r_slices[i] += slice_out * t_norm_patch
                n_slices[i] += t_norm_patch
            batch_cnt = 0
            r_slices = list()
            i_slices = list()
            n_slices = list()

    result = t_result
    norm_map = t_norm_map

    final_result = np.squeeze(finalize(result, norm_map, use_cuda_kernels=use_cuda_kernels), axis=0)
    return final_result, subvol_cnt


def infer_single_query_full(runner: EngineRunner,
                            query: np.ndarray,
                            dtypestr: str,
                            fmtstr: str,
                            batch_size: int = 1,
                            zero_pad: bool = False,
                            use_cuda_kernels: bool = False) -> Tuple[np.ndarray, int]:
    """
    Performs inference upon data and summarize work with mystr
    Done on full image (not sliding window)
    Parameters
    ----------
        runner:
            Runner running TRT engine
        query: numpy.ndarray
            KiTS19 image to infer
        dtypestr: str
            Input dtype in string
        fmtstr: str
            Input format in string
        batch_size: int
            Batch size to use for sub-volume inference
        zero_pad: bool
            Whether to zero-pad input on-the-fly
        use_cuda_kernels: bool
            Whether to use custom CUDA kernels
    """

    # Inference output format is LINEAR
    # 1ch FP16/INT8 input all appear to be LINEAR
    assert not zero_pad, "full image inference only tested with LINEAR input"
    assert not use_cuda_kernels, "CUDA kernels were written for sliding window inference only"
    output_shape = [1, 3] + list(runner.engine.get_binding_shape(0)[2:])
    image = query[np.newaxis, ...]

    if batch_size > 1:
        batch_size = 1

    runner_output = runner([image], batch_size)
    result = runner_output[0].reshape(output_shape)
    final_result = np.squeeze(apply_argmax(result), axis=0)

    return final_result, 1


def to_one_hot(my_array: np.ndarray) -> np.ndarray:
    """
    Changes class information into one-hot encoded information
    Number of classes in KiTS19 is 3: background, kidney segmentation, tumor segmentation
    As a result, 1 channel of class info turns into 3 channels of one-hot info
    """
    my_array = prepare_one_hot(my_array, num_classes=3)
    my_array = np.transpose(my_array, (0, 4, 1, 2, 3)).astype(np.float64)
    return my_array


def prepare_one_hot(my_array: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Reinterprets my_array into one-hot encoded, for classes as many as num_classes
    """
    res = np.eye(num_classes)[np.array(my_array).reshape(-1)]
    return res.reshape(list(my_array.shape) + [num_classes])


def get_dice_score(case: str, prediction: np.ndarray, target: np.ndarray) -> Tuple[str, float]:
    """
    Calculates DICE score of prediction against target, for classes as many as case
    One-hot encoded form of case/prediction used for easier handling
    Background case is not important and hence removed
    """
    # constants
    channel_axis = 1
    reduce_axis = (2, 3, 4)
    smooth_nr = 1e-6
    smooth_dr = 1e-6

    # apply one-hot
    prediction = to_one_hot(prediction)
    target = to_one_hot(target)

    # remove background
    target = target[:, 1:]
    prediction = prediction[:, 1:]

    # calculate dice score
    assert target.shape == prediction.shape, \
        f"Different shape -- target: {target.shape}, prediction: {prediction.shape}"
    assert target.dtype == np.float64 and prediction.dtype == np.float64, \
        f"Unexpected dtype -- target: {target.dtype}, prediction: {prediction.dtype}"

    # intersection for numerator; target/prediction sum for denominator
    # easy b/c one-hot encoded format
    intersection = np.sum(target * prediction, axis=reduce_axis)
    target_sum = np.sum(target, axis=reduce_axis)
    prediction_sum = np.sum(prediction, axis=reduce_axis)

    # get DICE score for each class
    dice_val = (2.0 * intersection + smooth_nr) / \
        (target_sum + prediction_sum + smooth_dr)

    # return after removing batch dim
    return (case, dice_val[0])


def load_pickled_outputs(output_list: List, temp_dir: Union[str, os.PathLike]) -> Dict:
    predictions = dict()
    for case in output_list:
        with open(Path(temp_dir, case + ".pkl"), 'rb') as f:
            case_d = pickle.load(f)
        predictions.update({case: case_d})
    return predictions


def evaluate(target_files: List,
             preprocessed_data_dir: Union[str, os.PathLike],
             postprocessed_data_dir: Union[str, os.PathLike],
             num_proc: int) -> None:
    """
    Collects and summarizes DICE scores of all the predicted files using multi-processes
    """
    bundle = list()

    for case in target_files:
        groundtruth_path = Path(preprocessed_data_dir,
                                "nifti", case, "segmentation.nii.gz").absolute()
        prediction_path = Path(postprocessed_data_dir,
                               case, "prediction.nii.gz").absolute()

        groundtruth = nib.load(groundtruth_path).get_fdata().astype(np.uint8)
        prediction = nib.load(prediction_path).get_fdata().astype(np.uint8)

        groundtruth = np.expand_dims(groundtruth, 0)
        prediction = np.expand_dims(prediction, 0)

        assert groundtruth.shape == prediction.shape,\
            "{} -- groundtruth: {} and prediction: {} have different shapes".format(
                case, groundtruth.shape, prediction.shape)

        bundle.append((case, groundtruth, prediction))

    with Pool(num_proc) as p:
        dice_scores = p.starmap(get_dice_score, bundle)

    save_evaluation_summary(postprocessed_data_dir, dice_scores)


def save_evaluation_summary(postprocessed_data_dir: Union[str, os.PathLike],
                            dice_scores: Iterable) -> None:
    """
    Stores collected DICE scores in CSV format: $(POSTPROCESSED_DATA_DIR)/summary.csv
    """
    sum_path = Path(postprocessed_data_dir, "summary.csv").absolute()
    df = pd.DataFrame()

    for _s in dice_scores:
        case, arr = _s
        kidney = arr[0]
        tumor = arr[1]
        composite = np.mean(arr)
        df = df.append(
            {
                "case": case,
                "kidney": kidney,
                "tumor": tumor,
                "composite": composite
            }, ignore_index=True)

    df.set_index("case", inplace=True)
    # consider NaN as a crash hence zero
    df.loc["mean"] = df.fillna(0).mean()

    df.to_csv(sum_path)


def save_nifti(bundle: List) -> None:
    """
    Saves single segmentation result from inference into NIFTI file
    """
    # Note that affine has to be valid, otherwise NIFTI image will look weird
    image, affine, path_to_file = bundle
    if len(image.shape) != 3:
        assert len(image.shape) == 4 and image.shape[0] == 1,\
            "Unexpected image: {}".format(image.shape)
        image = np.squeeze(image, 0)
    nifti_image = nib.Nifti1Image(image, affine=affine)
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_image, path_to_file)


def save_predictions(predictions: Dict,
                     output_dir: Union[str, os.PathLike],
                     preprocessed_data_dir: Union[str, os.PathLike],
                     num_proc: int) -> None:
    """
    Saves all the segmentation result from inference into NIFTI files using affine matrices
    Affine matrices were stored for input images during preprocessing
    NIFTI files stored as $(POSTPROCESSED_DATA)/KiTS19/reference/case_XXXX/prediction.nii.gz
    """
    with open(Path(preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
        preprocessed_files_content = pickle.load(f)
    aux = preprocessed_files_content['cases']

    bundle = list()
    for case, case_d in predictions.items():
        pred_file_path = Path(output_dir, case, "prediction.nii.gz")
        bundle.append((case_d, aux[case]['reshaped_affine'], pred_file_path))

    with Pool(num_proc) as p:
        p.map(save_nifti, bundle)

    p.join()
    p.close()


def run_3dunet_accuracy(engine_file: str,
                        batch_size: int,
                        num_samples: int,
                        verbose: bool = False,
                        preconditioned: bool = False,
                        full_img_inference: bool = False,
                        zero_pad: bool = False,
                        use_cuda_kernels: bool = False) -> float:
    """
    Calculate accuracy and return the DICE score (composite score) of KiTS19 samples used

    Returns:
        float:
            DICE score

    Arguments:
        --engine_file       : File to load engine from
        --batch_size        : Batch size to use for the engine
        --num_samples        : Number of samples to use for accuracy runner
        --verbose           : Print detailed information
        --preconditioned    : Use preconditioned Gaussian patches
        --full_img_inference: Perform inference on full image; else sliding window inference
        --zero_pad          : Zero-pad LINEAR input to sliding window to 32Ch NC/32DHW32 format, on-the-fly
        --use_cuda_kernels  : Use custom CUDA kernels for sliding window inference
    """

    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)
    num_proc = 8

    if verbose:
        logging.info("Running UNET accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    batch_size: {:}".format(batch_size))
        logging.info("    num_samples: {:}".format(num_samples))
        logging.info("    engine_input_dtype: {:}".format(input_dtype))
        logging.info("    engine_input_format: {:}".format(input_format))

    format_string = "Unknown"
    dtype_string = "Unknown"
    if input_dtype == trt.DataType.FLOAT:
        dtype_string = "float32"
        format_string = "fp32"
    elif input_dtype == trt.DataType.INT8:
        dtype_string = "int8"
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8"
        elif input_format == trt.TensorFormat.CDHW32:
            format_string = "int8_cdhw32"
    elif input_dtype == trt.DataType.HALF:
        dtype_string = "float16"
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "fp16"
        elif input_format == trt.TensorFormat.DHWC8:
            format_string = "fp16_dhwc8"
    assert format_string != "Unknown" or dtype_string != "Unknown",\
        "Unsupported data type and format combination"

    preproc_dir_env = os.getenv("PREPROCESSED_DATA_DIR", default="build/preprocessed_data")
    postproc_dir_env = os.getenv("POSTPROCESSED_DATA_DIR", default="build/postprocessed_data")
    raw_dir = Path(preproc_dir_env, "KiTS19", "reference")
    image_dir = Path(preproc_dir_env, "KiTS19", "inference", format_string)
    if zero_pad:
        assert format_string == "int8_cdhw32", "INT8 CDHW32 engine is required, for --zero_pad_int8_linear_input"
        print("Using INT8 LINEAR input files, that are to be sliced and zero-padded to cdhw32")
        # redirecting input to be INT8 LINEAR
        image_dir = Path(preproc_dir_env, "KiTS19", "inference", "int8")
    output_dir = Path(postproc_dir_env, "KiTS19", "prediction")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("build", "temporary")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    # 43 is max number of KiTS19 inference dataset
    if num_samples is None or num_samples <= 0:
        num_samples = 43

    image_list = list()
    output_list = list()
    with open("data_maps/kits19/val_map.txt") as f:
        for line in f:
            image_list.append(line.split()[0])

    if full_img_inference:
        print("WARNING > Full image inference only tested for inputs of 1ch 192x384x384")
        # image list of dim: 192x384x384
        image_list = [
            "case_00000",
            "case_00003",
            "case_00049",
            "case_00052",
            "case_00078",
            "case_00157",
            "case_00161",
            "case_00206",
        ]

    assert num_samples > 0 and any(image_list), "Error in populating input images"
    num_samples = min(num_samples, len(image_list))

    single_query_infer_func = infer_single_query_full if full_img_inference else\
        infer_single_query_preconditioned if preconditioned else\
        infer_single_query

    for image_idx in range(0, num_samples):
        img_name = image_list[image_idx]
        output_list.append(img_name)
        this_img = np.load(Path(image_dir, img_name + ".npy"))

        mystr = "{:}/{:} -- {:} with shape = {:}".format(
            image_idx + 1, num_samples, img_name, this_img.shape)

        start_time = time.time()
        outputs, subvol_cnt = single_query_infer_func(runner, this_img, dtype_string, format_string,
                                                      batch_size, zero_pad, use_cuda_kernels)
        end_time = time.time()

        mystr += ", {:3} sub-volumes".format(subvol_cnt)
        if verbose:
            logging.info("{}, took {:f} sec".format(mystr, end_time - start_time))

        with open(Path(temp_dir, img_name + ".pkl"), 'wb') as f:
            pickle.dump(outputs, f)

    predictions = load_pickled_outputs(output_list, temp_dir)
    save_predictions(predictions, output_dir, raw_dir, num_proc)
    evaluate(output_list, raw_dir, output_dir, num_proc)

    dice_score = 0.0
    df = pd.read_csv(Path(output_dir, "summary.csv"))
    final = df.loc[df['case'] == 'mean']
    composite = float(final['composite'])
    kidney = float(final['kidney'])
    tumor = float(final['tumor'])
    if verbose:
        logging.info("Accuracy: mean = {:.5f}, kidney = {:.4f}, tumor = {:.4f}".format(
            composite, kidney, tumor))
    dice_score = composite

    return dice_score


def parse_myargs() -> argparse.Namespace:
    """
    Handles arguments, used in this script

    Returns:
        argparse.Namespace:
            Namespace populated with argument strings and associated attributes

    Arguments:
        --engine_file               : File to load engine from
        --batch_size                : Batch size to use for the engine
        --num_samples               : Number of samples to use for accuracy runner
        --verbose                   : If provided, verbose output is printed out
        --full_image                : If provided, performs inference on full images;
                                        otherwise performs sliding window inference
        --preconditioned            : Use Gaussian patches that are preconditioned and stored in the storage
        --zero_pad_int8_linear_input: Zero pad the LINEAR sliding window input to
                                        NC/32DHW32 sliding window input on-the-fly
        --use_cuda_kernels          : Use Custom CUDA kernels for slicing, patching, accumulating and ArgMax
    """

    PARSER = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    PARSER.add_argument("--engine_file",
                        required=True,
                        type=str,
                        help="File to load engine from")

    PARSER.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size to use for the engine")

    PARSER.add_argument("--num_samples",
                        type=int,
                        default=0,
                        help="Number of samples to use for accuracy runner")

    PARSER.add_argument("--verbose",
                        action="store_true",
                        help="Use verbose output")

    PARSER.add_argument("--full_image",
                        action="store_true",
                        help="If provided, performs inference on full images; else performs sliding window inference")

    PARSER.add_argument("--preconditioned",
                        action="store_true",
                        help="If provided, performs inference with preconditioned gaussian weight patch")

    PARSER.add_argument("--zero_pad_int8_linear_input",
                        action="store_true",
                        help="INT8 input and sliding window inference only; if provided, "
                             "use linear input and zero pad on-the-fly during sliding window inference")

    PARSER.add_argument("--use_cuda_kernels",
                        action="store_true",
                        help="Use kernels for doing sliding window inference")

    args = PARSER.parse_args()

    return args


def main():
    """
    Run accuracy test
    """
    logging.info("Running accuracy test...")

    args = parse_myargs()
    acc = run_3dunet_accuracy(args.engine_file,
                              args.batch_size,
                              args.num_samples,
                              verbose=args.verbose,
                              preconditioned=args.preconditioned,
                              full_img_inference=args.full_image,
                              zero_pad=args.zero_pad_int8_linear_input,
                              use_cuda_kernels=args.use_cuda_kernels)

    logging.info("Accuracy: {:}".format(acc))


if __name__ == "__main__":
    main()
