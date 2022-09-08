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

#pragma once

#include <vector>
#include <cassert>
#include <cstdint>
#include <cmath>

#include "suinfer_common.h"
#include "sugraph.h"

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))

namespace suinfer {

static constexpr float k_epsilon = 0.000001;

inline bool FloatEqual(float a, float b) {
  auto diff = a - b;
  return (-k_epsilon < diff and diff < k_epsilon);
}

class bf16 _SUINFER_FINAL {
public:
  DECLARE_CTORS_FULL(bf16);

  explicit bf16(float float_data) { _set(float_data); }

  explicit bf16(uint16_t uint16_data)
    : uint16_data_(uint16_data), float_data_(_Uint16ToFloat(uint16_data_)) {}

  bf16& operator=(const bf16& rhs) {
    if (this == &rhs) {
      return *this;
    }

    uint16_data_ = rhs.uint16_data_;
    float_data_ = rhs.float_data_;

    return *this;
  }

  bf16& operator=(bf16&& rhs) {
    if (this == &rhs) {
      return *this;
    }

    uint16_data_ = rhs.uint16_data_;
    rhs.uint16_data_ = 0;

    float_data_ = rhs.float_data_;
    rhs.float_data_ = 0.0;

    return *this;
  }

  uint16_t getUint16() const { return uint16_data_; }
  float getFloat() const { return float_data_; }

  bool operator==(const bf16& rhs) const {
    return (uint16_data_ == rhs.uint16_data_ && float_data_ == rhs.float_data_);
  }

private:
  static uint16_t _FloatToUint16(float float_data) {
    return (*reinterpret_cast<uint32_t*>(&float_data) >> 16);
  }

  static float _Uint16ToFloat(uint16_t uint16_data) {
    uint32_t uint32_data = (uint16_data << 16);
    return (*reinterpret_cast<float*>(&uint32_data));
  }

  void _set(float float_data) {
    uint16_data_ = _FloatToUint16(float_data);
    float_data_ = _Uint16ToFloat(uint16_data_);
  }

private:
  uint16_t uint16_data_ = 0;
  float float_data_ = 0.0;
};

class float16_t _SUINFER_FINAL {
public:
  float16_t() = default;

  explicit float16_t(uint16_t uint16_data) {
    uint16_data_ = uint16_data;
    float_data_ = _Uint16ToFloat(uint16_data_);
  }

  float16_t(const float16_t& rhs) { *this = rhs; }

  float16_t& operator=(const float16_t& rhs) {
    if (this == &rhs) {
      return *this;
    }

    uint16_data_ = rhs.uint16_data_;
    float_data_ = rhs.float_data_;

    return *this;
  }

  float16_t(float16_t&& rhs) { *this = std::move(rhs); }

  float16_t& operator=(float16_t&& rhs) {
    if (this == &rhs) {
      return *this;
    }

    uint16_data_ = rhs.uint16_data_;
    rhs.uint16_data_ = 0;

    float_data_ = rhs.float_data_;
    rhs.float_data_ = 0.0;

    return *this;
  }

  bool operator==(const float16_t& rhs) const {
    return (uint16_data_ == rhs.uint16_data_ && float_data_ == rhs.float_data_);
  }

  uint16_t getUint16() const { return uint16_data_; }
  float getFloat() const { return float_data_; }

private:
  static float _Uint16ToFloat(uint16_t uint16_data) {
    uint32_t uint32_data = uint32_t(uint16_data & 0x03FF) << 13;
    uint32_t exponent = uint32_t(uint16_data & 0x7C00) >> 10;

    if (exponent != 0) {
      // add the exponent of the float, converting the offset binary formats of
      // the representations
      uint32_data |= (((exponent - 15 + 127) << 23) & 0x7F800000);
    }

    // add the sign bit.
    uint32_data |= uint32_t(uint16_data & 0x8000) << 16;
    return *(reinterpret_cast<float*>(&uint32_data));
  }

private:
  uint16_t uint16_data_ = 0;
  float float_data_ = 0.0;
};

inline int32_t int64_to_int32(int64_t i64) {
  if (i64 > (int64_t)(1 << 30)) {
    return (1 << 30);
  }

  if (i64 < (int64_t)(-(1 << 30))) {
    return (-(1 << 30));
  }

  return i64;
}

template<typename T1, typename T2>
inline void VectorConvert(std::vector<T1>& vec_dst,
                          const std::vector<T2>& vec_src) {
  vec_dst.resize(vec_src.size());

  for (std::size_t i = 0; i < vec_src.size(); ++i) {
    vec_dst[i] = static_cast<T1>(vec_src[i]);
  }
}

template<typename T1>
inline void VectorConvert(std::vector<T1>& vec_dst,
                          const std::vector<bf16>& vec_src) {
  vec_dst.resize(vec_src.size());

  for (std::size_t i = 0; i < vec_src.size(); ++i) {
    vec_dst[i] = vec_src[i].getFloat();
  }
}

template<typename T>
static inline std::size_t GetFlatSize(const T& shape, std::size_t shape_size,
                                      std::size_t start = 0) {
  std::size_t size = 1;
  for (std::size_t i = start; i < shape_size; ++i) {
    size *= shape[i];
  }
  return size;
}

static inline std::size_t GetFlatSize(
  const std::vector<sugraph::Shape::dim_t>& shape, std::size_t start = 0) {
  return GetFlatSize(shape, shape.size(), start);
}

// Indexes matches shape style, high end is lower dimension
// E.g. shape = [1,2,3,4], indexes = [4,3,2,1], flat_index =
// 4*(2*3*4)+3*(3*4)+2*(4)+1
static inline std::size_t IndexToFlatIndex(
  const std::vector<sugraph::Shape::dim_t>& shape,
  const std::vector<sugraph::Shape::dim_t>& indexes) {
  assert(shape.size() == indexes.size());
  std::size_t ret = 0;
  std::size_t i = 0;
  for (auto index : indexes) {
    assert(shape[i] > 0);
    ret *= shape[i++];
    ret += index;
  }
  return ret;
}

static inline std::vector<sugraph::Shape::dim_t> FlatIndexToIndex(
  const std::vector<sugraph::Shape::dim_t>& shape, std::size_t index) {
  std::vector<sugraph::Shape::dim_t> ret;
  ret.resize(shape.size());

  std::size_t remain = index;
  for (std::size_t i = 0; i < shape.size(); i++) {
    auto dindex = shape.size() - i - 1;
    sugraph::Shape::dim_t d = shape[dindex];
    assert(d > 0);
    std::size_t num = remain % d;
    remain /= d;

    ret[dindex] = (sugraph::Shape::dim_t)num;
  }
  return ret;
}

static inline std::vector<sugraph::Shape::dim_t> PermuteShapeOrIndex(
  const std::vector<sugraph::Shape::dim_t>& src,
  const std::vector<sugraph::Shape::dim_t>& dst_shape_idx) {
  assert(src.size() == dst_shape_idx.size());

  std::vector<sugraph::Shape::dim_t> dst;
  dst.resize(dst_shape_idx.size());

  for (std::size_t i = 0; i < dst_shape_idx.size(); i++) {
    dst[i] = src[dst_shape_idx[i]];
  }

  return dst;
}

template<typename T>
static inline std::vector<T> Permute(
  const std::vector<T>& tensor,
  const std::vector<sugraph::Shape::dim_t>& src_shape,
  const std::vector<sugraph::Shape::dim_t>& dst_shape_idx) {
  assert(src_shape.size() == dst_shape_idx.size());

  std::vector<T> ret;
  ret.resize(GetFlatSize(src_shape));

  std::vector<sugraph::Shape::dim_t> dst_shape =
    PermuteShapeOrIndex(src_shape, dst_shape_idx);

  for (std::size_t src_i = 0; src_i < tensor.size(); src_i++) {
    // First get index in src shape, then get corresponding index in dst shape
    std::vector<sugraph::Shape::dim_t> src_index =
      FlatIndexToIndex(src_shape, src_i);

    // Generate corresponding dst index
    std::vector<sugraph::Shape::dim_t> dst_index =
      PermuteShapeOrIndex(src_index, dst_shape_idx);

    // Move data
    std::size_t dst_i = IndexToFlatIndex(dst_shape, dst_index);
    ret[dst_i] = tensor[src_i];
  }

  return ret;
}

template<typename T>
static inline std::vector<T> Pad(
  const std::vector<T>& tensor,
  const std::vector<sugraph::Shape::dim_t>& src_shape,
  const std::vector<sugraph::Shape::dim_t>& dst_shape) {
  assert(src_shape.size() == dst_shape.size());

  std::vector<T> ret;
  ret.resize(GetFlatSize(dst_shape), 0);

  for (std::size_t src_i = 0; src_i < tensor.size(); src_i++) {
    // First get index in src shape, then get corresponding index in dst shape
    std::vector<sugraph::Shape::dim_t> src_index =
      FlatIndexToIndex(src_shape, src_i);

    // dst index is same as src index
    // Move data
    std::size_t dst_i = IndexToFlatIndex(dst_shape, src_index);
    ret[dst_i] = tensor[src_i];
  }

  return ret;
}

static inline float GetMold(const float* nums, std::size_t cnt) {
  float sum = 0;

  for (std::size_t i = 0; i < cnt; ++i) {
    auto num = nums[i];
    sum += (num * num);
  }

  return std::sqrt(sum);
}

inline float GetSimilarity(const float* nums_l, const float* nums_r,
                           std::size_t cnt) {
  float sum = 0;

  for (std::size_t i = 0; i < cnt; ++i) {
    sum += (nums_l[i] * nums_r[i]);
  }

  return (sum / (GetMold(nums_l, cnt) * GetMold(nums_r, cnt)));
}

} // namespace suinfer
