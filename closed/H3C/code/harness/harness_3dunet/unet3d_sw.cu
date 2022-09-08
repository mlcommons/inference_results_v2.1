/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "unet3d_sw.cuh"

namespace lwis
{

__global__ void UNet3DKiTS19SliceKernelFP32Linear(
    const float* __restrict__ d_in, float* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw)
    {
        d_out[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w]
            = d_in[p.image_w * p.image_h * (p.offset_d + d) + p.image_w * (p.offset_h + h) + (p.offset_w + w)];
    }
}

__global__ void UNet3DKiTS19SliceKernelI8Linear(
    const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw)
    {
        d_out[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w]
            = d_in[p.image_w * p.image_h * (p.offset_d + d) + p.image_w * (p.offset_h + h) + (p.offset_w + w)];
    }
}

__global__ void UNet3DKiTS19SliceKernelI8CDHW32(
    const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw)
    {
        d_out[32 * (p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w)]
            = d_in[p.image_w * p.image_h * (p.offset_d + d) + p.image_w * (p.offset_h + h) + (p.offset_w + w)];
    }
}

__global__ void UNet3DKiTS19PatchKernel(const __half* __restrict__ d_in, const __half* __restrict__ d_patch,
    __half* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    if (d < p.roi_dhw && h < p.roi_dhw && w < p.roi_dhw)
    {
#pragma unroll
        for (int c = 0; c < p.out_ch; ++c)
        {
            d_out[p.image_h * p.image_w * (p.offset_d + d) + p.image_w * (p.offset_h + h) + (p.offset_w + w)
                + p.image_size * c]
                += d_in[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w + p.roi_size * c]
                * d_patch[p.roi_dhw * p.roi_dhw * d + p.roi_dhw * h + w];
        }
    }
}

__global__ void UNet3DKiTS19ArgMaxKernel(
    const __half* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    __half a = d_in[p.image_h * p.image_w * d + p.image_w * h + w];
    __half b = d_in[p.image_h * p.image_w * d + p.image_w * h + w + p.image_size];
    __half c = d_in[p.image_h * p.image_w * d + p.image_w * h + w + 2 * p.image_size];
    __half m = b;
    uint8_t l = 1;
    if (a > b)
    {
        m = a;
        l = 0;
    }
    if (d < p.image_d && h < p.image_h && w < p.image_w)
    {
        d_out[p.image_h * p.image_w * d + p.image_w * h + w] = m > c ? l : 2;
    }
}

void UNet3DKiTS19SliceKernelFP32Linear_wrapper(
    void* d_in, void* d_out, const UNet3DParams* p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for slicing
    dim3 dimBlock_slice(p->roi_dhw, 1, 1);
    dim3 dimGrid_slice(p->roi_dhw, p->roi_dhw, 1);
    UNet3DKiTS19SliceKernelFP32Linear<<<dimGrid_slice, dimBlock_slice, 0, stream>>>(
        static_cast<float*>(d_in), static_cast<float*>(d_out), *p);
}

void UNet3DKiTS19SliceKernelI8Linear_wrapper(
    void* d_in, void* d_out, const UNet3DParams* p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for slicing
    dim3 dimBlock_slice(p->roi_dhw, 1, 1);
    dim3 dimGrid_slice(p->roi_dhw, p->roi_dhw, 1);
    UNet3DKiTS19SliceKernelI8Linear<<<dimGrid_slice, dimBlock_slice, 0, stream>>>(
        static_cast<int8_t*>(d_in), static_cast<int8_t*>(d_out), *p);
}

void UNet3DKiTS19SliceKernelI8CDHW32_wrapper(
    void* d_in, void* d_out, const UNet3DParams* p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for slicing
    dim3 dimBlock_slice(p->roi_dhw, 1, 1);
    dim3 dimGrid_slice(p->roi_dhw, p->roi_dhw, 1);
    UNet3DKiTS19SliceKernelI8CDHW32<<<dimGrid_slice, dimBlock_slice, 0, stream>>>(
        static_cast<int8_t*>(d_in), static_cast<int8_t*>(d_out), *p);
}

void UNet3DKiTS19PatchKernel_wrapper(void* d_in, void* d_patch, void* d_out, const UNet3DParams* p,
    const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for Gaussian patching & accumulating
    dim3 dimBlock_patch(p->roi_dhw, 1, 1);
    dim3 dimGrid_patch(p->roi_dhw, p->roi_dhw, 1);
    UNet3DKiTS19PatchKernel<<<dimGrid_patch, dimBlock_patch, 0, stream>>>(
        static_cast<__half*>(d_in), static_cast<__half*>(d_patch), static_cast<__half*>(d_out), *p);
}

void UNet3DKiTS19ArgMaxKernel_wrapper(
    void* d_in, void* d_out, const UNet3DParams* p, const cudaStream_t stream = 0, const int deviceId = 0)
{
    cudaSetDevice(deviceId);
    // for final ArgMax
    dim3 dimBlock_argmax(p->image_w, 1, 1);
    dim3 dimGrid_argmax(p->image_d, p->image_h, 1);
    UNet3DKiTS19ArgMaxKernel<<<dimGrid_argmax, dimBlock_argmax, 0, stream>>>(
        static_cast<__half*>(d_in), static_cast<int8_t*>(d_out), *p);
}

} // namespace lwis