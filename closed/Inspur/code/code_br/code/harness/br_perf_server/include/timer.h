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

#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>

#include "system_under_test.h"

// For debugging the timing of each part
class Timer {
public:
  Timer(const std::string& tag_) : tag(tag_) { std::cout << "Timer " << tag << " created." << std::endl; }
  void add(const std::chrono::duration<double, std::milli>& in) {
    ++count;
    total += in;
  }
  ~Timer() {
    std::cout << "Timer " << tag << " reports " << total.count() / count << " ms per call for " << count << " times."
              << std::endl;
  }

private:
  std::string tag;
  std::chrono::duration<double, std::milli> total{0};
  std::size_t count{0};
};

#define TIMER_ON 0

#if TIMER_ON
#define TIMER_START(s)       \
  static Timer timer##s(#s); \
  auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_END(s)
#endif

namespace brPerfServer {

class PerfTimer {
public:
  explicit PerfTimer(const char* op, std::size_t device_idx)
    : op_(op), device_idx_(device_idx), k_start_(std::chrono::high_resolution_clock::now()) {
    char buff[60] = {0};
    printTime(k_start_, buff, 33);
    std::stringstream ss;
    ss << buff << "      " << op_ << " starts on " << device_idx_ << std::endl;
    std::cout << ss.str();
  }

  ~PerfTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    char buff[60] = {0};
    printTime(end, buff, 33);
    std::stringstream ss;
    ss << buff << "      " << op_ << " ends on " << device_idx_
       << ". duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - k_start_).count() << std::endl;
    std::cout << ss.str();
    fflush(stdout);
  }

private:
  std::string op_;
  const std::size_t device_idx_;
  const std::chrono::time_point<std::chrono::high_resolution_clock> k_start_;

  void printTime(const std::chrono::high_resolution_clock::time_point& p, char* buff, std::size_t size);
};

// Util functions
std::vector<uint32_t> DecodeResult(const char* predict, std::size_t predict_size, suinfer::sugraph::DataLayout layout,
                                   std::size_t batch, std::size_t class_num = 1000);
std::vector<std::vector<uint32_t>> DecodeResult(const char* predict, std::size_t predict_size,
                                                std::size_t compute_unit_num, suinfer::sugraph::DataLayout layout,
                                                std::size_t batch, std::size_t class_num = 1000);
std::vector<std::string> ParseFileList(const char* arg);

} // namespace brPerfServer

namespace Biren {
namespace packer {
struct PackPair {
  int count;
  std::vector<int> pack;
};

typedef std::vector<PackPair> PackList;

template<typename T>
std::vector<int> histogram(const std::vector<T>& sequences, int max_sequence_length) {
  std::vector<int> sequence_lengths(max_sequence_length);
  for (const auto& seq : sequences) {
    assert(seq >= 1);
    sequence_lengths[seq - 1] += 1;
  }

  return sequence_lengths;
}

template<typename T>
struct PackedSequence {
  PackedSequence() {}
  PackedSequence(std::size_t _id, std::vector<mlperf::QuerySample*>&& _samples, std::vector<T>&& _masks)
    : id(_id), samples(std::move(_samples)), masks(std::move(_masks)) {}
  PackedSequence(PackedSequence&& pack) : samples(std::move(pack.samples)), masks(std::move(pack.masks)) {}

  PackedSequence& operator=(PackedSequence&& pack) noexcept {
    if (&pack == this) {
      return *this;
    }

    samples = std::move(pack.samples);
    masks = std::move(pack.masks);
  }

  std::size_t id{100};
  std::vector<mlperf::QuerySample*> samples;
  // This is the fourth input and this will be used to get correct response data.
  std::vector<T> masks;
};

template<typename T>
using PackedSequenceList = std::vector<PackedSequence<T>*>;

template<typename T>
struct Sequence {
  Sequence(mlperf::QuerySample* sample_, T len) : sample(sample_), length(len) {}
  mlperf::QuerySample* sample;
  T length;
};

template<typename T>
using SequenceList = std::vector<Sequence<T>>;

using LenToSampleMap = std::unordered_map<std::size_t, std::vector<mlperf::QuerySample*>>;

template<typename T>
PackedSequence<T> pack_one_instance(const SequenceList<T>& multi_sequence, int max_sequence_length) {
  PackedSequence<T> packed;
  auto offset = 0;
  auto sequence_index = 1; // used in the input mask
  packed.samples.reserve(multi_sequence.size());
  packed.masks.reserve(multi_sequence.size() * 2);
  for (const auto& seq : multi_sequence) {
    packed.samples.push_back(seq.sample);
    packed.masks.push_back(offset);
    offset += seq.length;
    packed.masks.push_back(offset);
  }
  return packed;
}

template<typename T>
PackedSequenceList<T> pack(LenToSampleMap& examples_by_length, PackList& packList, int max_sequence_length,
                           int max_sequences_per_pack) {
  PackedSequenceList<T> packedSeqs;
  std::vector<int> slice_offsets(max_sequence_length + 1);
  std::vector<int> slice_starts(max_sequences_per_pack);
  for (const auto& pl : packList) {
    for (auto j = 0; j < pl.pack.size(); ++j) {
      auto seq_len = pl.pack[j];
      slice_starts[j] = slice_offsets[seq_len];
      slice_offsets[seq_len] += pl.count;
    }
    for (auto i = 0; i < pl.count; ++i) {
      SequenceList<T> multi_sequence;
      for (auto j = 0; j < pl.pack.size(); ++j) {
        auto seq_len = pl.pack[j];
        multi_sequence.emplace_back(examples_by_length[seq_len][slice_starts[j] + i], seq_len);
      }
      auto* p = new PackedSequence<T>(pack_one_instance(multi_sequence, max_sequence_length));
      packedSeqs.emplace_back(p);
    }
  }

  return packedSeqs;
}
} // namespace packer
} // namespace Biren
