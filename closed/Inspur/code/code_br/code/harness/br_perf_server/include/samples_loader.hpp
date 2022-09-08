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
#include <cassert>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <deque>
#include <vector>

#include <besu.h>

#include <glog/logging.h>

#include "query_sample_library.h"
#include "test_settings.h"
#include "utils.hpp"

namespace samplesLoader {

using DataTransformer = std::function<void(char*, std::size_t)>;

class BrPerfSampleLoader : public mlperf::QuerySampleLibrary {
public:
  virtual std::vector<void*> GetSampleAddress(mlperf::QuerySampleIndex sample_index, std::size_t input_idx,
                                              std::size_t device_idx = 0) = 0;
  virtual std::size_t GetSampleSize(std::size_t input_idx) const = 0;
  virtual std::size_t GetSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept = 0;
  virtual std::vector<std::size_t>& GetAllSampleLenOfInputRef(std::size_t input_idx) noexcept = 0;
  virtual std::vector<std::size_t> GetAllSampleLenOfInput(std::size_t input_idx) noexcept = 0;
  virtual std::vector<std::vector<std::size_t>> getDimSizes(std::size_t input_idx) const = 0;
  virtual void setDataTranformerForInput(std::size_t input_idx, DataTransformer tranformer) noexcept = 0;
  virtual const std::string getFilePath(const std::string& raw_file_path) const = 0;
  virtual std::vector<char> readFile(const std::string& file_path) const = 0;
};

template<typename T>
std::size_t getDataLenWithoutLastZeros(void* data, std::size_t bytes) noexcept {
  T* data_ptr = reinterpret_cast<T*>(data);
  assert(bytes % sizeof(T) == 0);
  std::size_t total_data_len = bytes / sizeof(T) - 1;
  for (; total_data_len >= 0; --total_data_len) {
    if (*(data_ptr + total_data_len) != 0) {
      break;
    }
  }
  return total_data_len + 1;
}

class AppenedSampleLoader : public BrPerfSampleLoader {
public:
  AppenedSampleLoader(std::string name, std::string map_file_path /*the result value map file path*/,
                       std::vector<std::string> inputs_path /*data folders path, one for each input*/,
                       const std::string& data_file_suffix, std::size_t perf_sample_count /**/,
                       const std::vector<std::vector<std::size_t>>& dim_size, bool trim_last_zeros,
                       const std::vector<bool>& start_from_device, std::size_t device_idx, std::size_t padding = 0)
    : name_(name),
      perf_sample_count_(perf_sample_count),
      perf_sample_padding_(padding),
      map_file_path_(map_file_path),
      data_file_suffix_(data_file_suffix),
      dim_sizes_(dim_size),
      inputs_path_(inputs_path),
      trim_last_zeros_(trim_last_zeros),
      start_from_device_(start_from_device),
      device_idx_(device_idx) {
    // Get input size and allocate memory
    num_inputs_ = inputs_path_.size();
    sample_memory_.resize(num_inputs_);
    for (std::size_t idx = 0; idx < num_inputs_; ++idx) {
      auto& dims = dim_sizes_[idx];
      sample_size_[idx] = std::accumulate(dims.begin(), dims.end(), 0);
    }

    besuSetDevice(device_idx_);
    PRINT_CARD_ID(device_idx_);

    // Get number of samples
    // load and read in the sample map
    std::ifstream fs(map_file_path_);
    CHECK(fs) << "Unable to open sample map file: " << map_file_path_;

    char s[1024];
    while (fs.getline(s, 1024)) {
      std::istringstream iss(s);
      std::vector<std::string> r((std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

      file_label_map_.insert(
        std::make_pair(sample_count_, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
      ++sample_count_;
    }

    // as a safety, don't allow the perf_sample_count to be larger than sampleCount.
    perf_sample_count_ = std::min(perf_sample_count_, sample_count_);

    for (std::size_t input_idx = 0; input_idx < num_inputs_; input_idx++) {
      auto dim_size = dim_sizes_[input_idx].size();
      sample_memory_[input_idx].resize(dim_size);
      for (std::size_t idx = 0; idx < dim_size; ++idx) {
        getMallocFun(start_from_device_[input_idx])(
          &sample_memory_[input_idx][idx], (perf_sample_count_ + perf_sample_padding_) * dim_sizes_[input_idx][idx]);
      }
    }
}

  void setDataTranformerForInput(std::size_t input_idx, DataTransformer tranformer) noexcept {
    data_tranformers_[input_idx] = tranformer;
  }

  const std::string& Name() const override { return name_; }
  std::size_t TotalSampleCount() override { return sample_count_; }
  std::size_t PerformanceSampleCount() override { return perf_sample_count_; }
  std::vector<std::vector<std::size_t>> getDimSizes(std::size_t input_idx) const override { return dim_sizes_; }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    std::cout << "LoadSamplesToRam samples.size: " << samples.size() << std::endl;
    // copy the samples into pinned memory
    std::cout << "Loading data for device " << device_idx_ << std::endl;
    std::size_t current_idx = 0;
    sample_lens_.resize(num_inputs_);
    for (auto& sizes : sample_lens_) {
      sizes.resize(TotalSampleCount(), -1);
    }

    for (std::size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
      auto& sampleId = samples[sampleIndex];
      samples_address_[sampleId] = InputData(num_inputs_);
      for (std::size_t input_idx = 0; input_idx < num_inputs_; input_idx++) {
        std::string path = inputs_path_[input_idx] + "/" + getFilePath(std::get<0>(file_label_map_[sampleId]));
        std::vector<char> data = readFile(path);
        const auto& dims = dim_sizes_[input_idx];
        auto dim_cnt = dims.size();

        auto itr = data_tranformers_.find(input_idx);
        if (itr != data_tranformers_.end()) {
          itr->second(data.data(), data.size());
        }

        auto& sample_datas = samples_address_[sampleId][input_idx];
        sample_datas.resize(dim_cnt);
        auto data_ptr = data.data();
        // The better way is to set data process function to get this len. Because we may want get len for different
        // data type.
        if (trim_last_zeros_) {
          sample_lens_[input_idx][sampleId] = getDataLenWithoutLastZeros<int32_t>(data_ptr, data.size());
        } else {
          sample_lens_[input_idx][sampleId] = data.size();
        }

        for (std::size_t dim_idx = 0; dim_idx < dim_cnt; ++dim_idx) {
          // The size of data to copy use the min of dims size and data size.
          std::size_t size_to_copy = std::min(dims[dim_idx], data.size());
          std::size_t offset = std::min(size_to_copy, dims[dim_idx]);
          // Even if the real size of data is less than dim size, we keep the data occupying same size memory.
          // The offset in qsl memory is calculated from dims size.
          auto sampleAddress = static_cast<int8_t*>(sample_memory_[input_idx][dim_idx]) + current_idx * offset;

          getMemcpyFun(start_from_device_[input_idx])((char*)sampleAddress, data_ptr, size_to_copy);
          sample_datas[dim_idx] = sampleAddress;

          // The real data_ptr offset change shall follow the offset in qsl.
          data_ptr += dims[dim_idx];
        }
      }
      ++current_idx;
    }
  }

  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    // due to the removal of freelisting this code is currently a check and not required for functionality.
    for (auto& sampleId : samples) {
      auto it = samples_address_.find(sampleId);
      CHECK(it != samples_address_.end()) << "Sample: " << sampleId << " not allocated properly";
      auto& sample_addresses = it->second;
      CHECK(!sample_addresses.empty()) << "Sample: " << sampleId << " not loaded";
      sample_addresses.pop_back();
      if (sample_addresses.empty()) {
        samples_address_.erase(it);
      }
    }
  }

  virtual const std::string getFilePath(const std::string& raw_file_path) const {
    return data_file_suffix_.empty() ? raw_file_path : raw_file_path + data_file_suffix_;
  }

  virtual std::vector<char> readFile(const std::string& file_path) const {
    std::vector<char> encoded;

    std::ifstream in(file_path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
      std::cout << "Error open input file" << std::endl;
      return encoded;
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    encoded.resize(size);
    in.read(encoded.data(), size);

    if (in.gcount() != size) {
      std::cout << "Error reading input file" << std::endl;
    }

    return encoded;
  }

  std::vector<void*> GetSampleAddress(mlperf::QuerySampleIndex sample_index, std::size_t input_idx,
                                      std::size_t device_idx = 0) {
    auto it = samples_address_.find(sample_index);
    CHECK(it != samples_address_.end()) << "Sample: " << sample_index << " missing from RAM";
    CHECK(input_idx <= it->second.size()) << "invalid input_idx";

    std::size_t dims_cnt = dim_sizes_[input_idx].size();
    std::vector<void*> inputs(dims_cnt, nullptr);
    for (std::size_t idx = 0; idx < dims_cnt; ++idx) {
      inputs[idx] = it->second[input_idx][idx];
    }
    return inputs;
  }

  std::size_t GetSampleSize(std::size_t input_idx) const {
    return (sample_size_.empty() ? 0 : sample_size_[input_idx]);
  }
  std::size_t GetSampleLen(std::size_t input_idx, std::size_t sample_id) const noexcept override {
    return sample_lens_[input_idx][sample_id];
  }
  std::vector<std::size_t>& GetAllSampleLenOfInputRef(std::size_t input_idx) noexcept override {
    return sample_lens_[input_idx];
  }
  std::vector<std::size_t> GetAllSampleLenOfInput(std::size_t input_idx) noexcept override {
    return sample_lens_[input_idx];
  }

  ~AppenedSampleLoader() {
    std::size_t input_idx = 0;
    for (auto input_memory : sample_memory_) {
      if (start_from_device_[input_idx]) {
        for (auto dim_memory : input_memory) {
          besuFree(dim_memory);
        }
      }
    }
  }

private:
  std::size_t num_inputs_{0};
  int num_devices_{1};

  const std::string name_;
  std::size_t perf_sample_count_{0};
  std::size_t perf_sample_padding_{0};
  std::string map_file_path_;

  std::vector<std::size_t> sample_size_{0};

  // One for each input of model.
  std::vector<std::string> inputs_path_;
  bool trim_last_zeros_{false};
  std::vector<bool> start_from_device_{};

  std::size_t sample_count_{0};

  // maps sampleId to <fileName, label>
  std::map<mlperf::QuerySampleIndex, std::tuple<std::string, std::size_t>> file_label_map_;
  const std::string data_file_suffix_;

  // One for each input of model.
  // dims of input.
  std::vector<std::vector<std::size_t>> dim_sizes_;

  std::size_t device_idx_;

  // datas of num_inputs of
  // sampleMemory[input_idx][dim_idx];
  using SampleData = std::vector<void*>;
  using InputData = std::vector<SampleData>;
  InputData sample_memory_;
  // Input -> sample len
  std::vector<std::vector<std::size_t>> sample_lens_;

  // map[sampleId] ==> all inputs
  // map[sampleId][input_idx] ==> one sample
  // map[sampleId][input_idx][dim_idx] ==> one dim data
  std::map<mlperf::QuerySampleIndex, InputData> samples_address_;
  std::map<std::size_t, DataTransformer> data_tranformers_;
};

using SampleLoaderPtr = std::shared_ptr<BrPerfSampleLoader>;

class UniverseSampleLoader : public mlperf::QuerySampleLibrary {
public:
  UniverseSampleLoader(const std::vector<SampleLoaderPtr> loaders) : loaders_(loaders){};
  const std::string& Name() const override { return loaders_[0]->Name(); }
  std::size_t TotalSampleCount() override { return loaders_[0]->TotalSampleCount(); }
  std::size_t PerformanceSampleCount() override { return loaders_[0]->PerformanceSampleCount(); }

  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto loader : loaders_) {
      loader->LoadSamplesToRam(samples);
    }
  }
  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
    for (auto loader : loaders_) {
      loader->UnloadSamplesFromRam(samples);
    }
  }

private:
  std::vector<SampleLoaderPtr> loaders_;
};

} // namespace samplesLoader
