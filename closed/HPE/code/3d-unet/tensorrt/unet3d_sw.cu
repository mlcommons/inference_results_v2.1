/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_fp16.h>
#include <stdint.h>

struct UNet3DParams
{
    int image_d;
    int image_h;
    int image_w;
    int image_size;
    int offset_d;
    int offset_h;
    int offset_w;
    int roi_dhw;
    int roi_size;
    int in_ch;
    int out_ch;

    UNet3DParams()
    {
        image_d = 256;
        image_h = 256;
        image_w = 256;
        image_size = 256 * 256 * 256;
        offset_d = 0;
        offset_h = 0;
        offset_w = 0;
        roi_dhw = 128;
        roi_size = 128 * 128 * 128;
        in_ch = 1;
        out_ch = 3;
    }
};

extern "C" __global__ void UNet3DKiTS19SliceKernelFP32Linear(
    const float* __restrict__ d_in, float* __restrict__ d_out, const UNet3DParams* p)
{
    const int d = blockIdx.x;
    const int h = blockIdx.y;
    const int w = threadIdx.x;

    d_out[p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w]
        = d_in[p->image_w * p->image_h * (p->offset_d + d) + p->image_w * (p->offset_h + h) + (p->offset_w + w)];
}

extern "C" __global__ void UNet3DKiTS19SliceKernelI8Linear(
    const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams* p)
{
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = threadIdx.x;

    d_out[p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w]
        = d_in[p->image_w * p->image_h * (p->offset_d + d) + p->image_w * (p->offset_h + h) + (p->offset_w + w)];
}

extern "C" __global__ void UNet3DKiTS19SliceKernelI8CDHW32(
    const int8_t* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams* p)
{
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = threadIdx.x;

    d_out[32 * (p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w)]
        = d_in[p->image_w * p->image_h * (p->offset_d + d) + p->image_w * (p->offset_h + h) + (p->offset_w + w)];
}

extern "C" __global__ void UNet3DKiTS19PatchKernel(const __half* __restrict__ d_in, const __half* __restrict__ d_patch,
    __half* __restrict__ d_out, const UNet3DParams* p)
{
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = threadIdx.x;

#pragma unroll
    for (int c = 0; c < p->out_ch; ++c)
    {
        d_out[p->image_h * p->image_w * (p->offset_d + d) + p->image_w * (p->offset_h + h) + (p->offset_w + w)
            + p->image_size * c]
            += d_in[p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w + p->roi_size * c]
            * d_patch[p->roi_dhw * p->roi_dhw * d + p->roi_dhw * h + w];
    }
}

extern "C" __global__ void UNet3DKiTS19ArgMaxKernel(
    const __half* __restrict__ d_in, int8_t* __restrict__ d_out, const UNet3DParams* p)
{
    int d = blockIdx.x;
    int h = blockIdx.y;
    int w = threadIdx.x;

    __half a = d_in[p->image_h * p->image_w * d + p->image_w * h + w];
    __half b = d_in[p->image_h * p->image_w * d + p->image_w * h + w + p->image_size];
    __half c = d_in[p->image_h * p->image_w * d + p->image_w * h + w + 2 * p->image_size];
    __half m = b;
    uint8_t l = 1;
    if (a > b)
    {
        m = a;
        l = 0;
    }
    d_out[p->image_h * p->image_w * d + p->image_w * h + w] = m > c ? l : 2;
}
