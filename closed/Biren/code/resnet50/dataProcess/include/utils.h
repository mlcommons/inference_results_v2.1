/*
 * Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
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

#include <chrono>
#include <sstream>

#include "sugraph.h"

using void_ptr = void*;

void write_file(const std::string& path, char* output, size_t len);
void read_file(const std::string& path, char* output, size_t len);

std::vector<std::string> ParseFileList(const char* arg);

template<size_t H, size_t W, size_t C, typename T = uint8_t>
void CHW2HWC(T* src, T* dst) {
  if (!dst or !src)
    return;

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int src_idx = c * H * W + h * W + w;
        int dst_idx = h * W * C + w * C + c;
        *(T*)(dst + dst_idx) = *(T*)(src + src_idx);
      }
    }
  }
}

template<size_t C, size_t H, size_t W, typename T = uint8_t>
void HWC2CHW(T* src, T* dst) {
  if (!dst or !src)
    return;

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        int src_idx = h * W * C + w * C + c;
        int dst_idx = c * H * W + h * W + w;
        *(T*)(dst + dst_idx) = *(T*)(src + src_idx);
      }
    }
  }
}

template<size_t C, size_t H, size_t W, size_t left = 0, size_t up = 0,
         size_t right = 0, size_t down = 0, typename T = uint8_t>
void pad(T* src, T* dst) {
  if (!dst or !src)
    return;

  for (size_t c = 0; c < C; ++c) {
    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        int src_idx = c * H * W + h * W + w;
        int dst_idx = c * (H + up + down) * (W + left + right) +
                      (h + up) * (W + left + right) + (w + left);
        *(T*)(dst + dst_idx) = *(T*)(src + src_idx);
      }
    }
  }
}

void generate_outputs(const std::string& filepath, uint8_t* output, size_t size,
                      const std::string& outpath);
void quantifer(float* data, size_t size, uint8_t* output);
