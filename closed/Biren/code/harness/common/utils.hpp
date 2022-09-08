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

#include <besu.h>
#include <numaif.h>
#include <pthread.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
#define FOCEINLINE __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

using BertInputDataType = int32_t;
using BertOutputDataType = _Float32;
constexpr std::size_t BertMaxSeqLength = 512;
// constexpr std::size_t BertMaxSeqLength = 384;
constexpr std::size_t BertMaxSeqPerBatch = 3;

constexpr std::size_t BertInputIdsIdx = 0;
constexpr std::size_t BertSegmentIdsIdx = 1;
constexpr std::size_t BertPosIdsIdx = 2;
constexpr std::size_t BertPosMaskIdx = 3;

// #define DEBUG

// Utilities in debug mode

#ifdef DEBUG
#define PRINT_CARD_ID(deviceidx)                            \
  {                                                         \
    BesuDeviceProperties bdp;                               \
    besuGetDeviceProperties(&bdp, deviceidx);               \
    std::cout << "bdp.cardId: " << bdp.cardId << std::endl; \
  }
#else
#define PRINT_CARD_ID(deviceidx)
#endif

namespace brPerfServer {
class PerfTimer;
}

#ifdef DEBUG
#define TIME(s, i) brPerfServer::PerfTimer _(s, i)
#define TIME_START(s, i) brPerfServer::PerfTimer _(s, i) {
#define TIME_END }
#else
#define TIME(s, i)
#define TIME_START(s, i)
#define TIME_END
#endif

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t next = 0;
  while (next != std::string::npos) {
    next = input.find(delimiter, start);
    result.emplace_back(input, start, next - start);
    start = next + 1;
  }
  return result;
}

inline std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

inline void GetAllDeviceCardIds(std::map<size_t, size_t>& out_devices) {
  uint32_t card_count{0};
  besuGetDeviceCount(&card_count);

  BesuDeviceProperties bdp;
  for (size_t device_idx = 0; device_idx < card_count; ++device_idx) {
    besuGetDeviceProperties(&bdp, device_idx);
    out_devices[bdp.cardId] = device_idx;
  }

#if defined(DEBUG)
  for (auto device : out_devices) {
    std::cout << "bdp.cardId: " << device.first << ", device_idx: " << device.second << std::endl;
  }
#endif
}

inline std::vector<int32_t> GetDevicesIdsFromString(std::string& device_ids) {
  std::vector<int32_t> devices;
  auto device_names = split(device_ids, ',');
  std::for_each(device_names.cbegin(), device_names.cend(),
                [&devices](const std::string& idx) { devices.emplace_back(std::stoi(idx)); });

  return devices;
}

inline int32_t GetEngineBatchSizeForModel(const std::string& model_name) {
  if (model_name.compare("bert") == 0) {
    return 1;
  } else if (model_name.compare("resnet50") == 0) {
    return 16;
  }
  return -1;
}

inline uint8_t GetBitNumbersFromMask(uint64_t mask) {
  uint8_t num = 0;
  for (int i = 0; (i < 64) && (0 != mask); ++i) {
    if (mask & 0x1) {
      num++;
    }
    mask = mask >> 1;
  }
  return num;
}

inline uint8_t GetHighestBitNumberFromMask(uint64_t mask) {
  uint8_t num = 0;
  for (int i = 63; (i >= 0); --i) {
    if (mask & (0x1LL << i)) {
      return i + 1;
    }
  }
  return 0;
}

inline uint8_t GetLowestBitNumberFromMask(uint64_t mask) {
  uint8_t num = 0;
  for (int i = 0; (i < 64); ++i) {
    if (mask & (0x1LL << i)) {
      return i + 1;
    }
  }
  return 0;
}

inline uint8_t GetSpcNumFromMask(uint64_t mask) {
  if (mask) {
    return GetLowestBitNumberFromMask(mask) - 1;
  }
  return 0;
}

inline uint32_t GetDeviceSPCMasks(uint32_t deviceIdx) {
  BesuDeviceProperties prop;
  BesuStatus ret = besuGetDeviceProperties(&prop, deviceIdx);
  if (ret != BESU_STATUS_SUCCESS) {
    std::cerr << "Failed to call besuGetDeviceProperties: " << ret << std::endl;
    return 0;
  }
  auto spcMask = prop.spcMask;
  return prop.spcMask[0];
}
using MallocFun = BesuStatus (*)(void**, uint64_t);
using MemcpyFun = BesuStatus (*)(void*, const void*, uint64_t);

inline MallocFun getMallocFun(bool start_from_device) {
  if (start_from_device) {
    return besuMallocDevice;
  } else {
    return besuMallocHost;
  }
}

inline MemcpyFun getMemcpyFun(bool /*start_from_device*/) { return besuMemcpy; }
