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
#include <chrono>
#include <deque>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include <besu.h>
#include "engine.h"
#include "loadgen.h"
#include "samples_loader.hpp"
#include "suinfer.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "utils.hpp"
#include "timer.h"

#include "queue_wrapper.h"

namespace brPerfServer {
using namespace std::chrono_literals;

class Server;
class Device;
struct ServerSettings;
struct Batch;

using BindingHandle = suinfer::backend::memoryHandle_t;
using DevicePtr_t = std::shared_ptr<Device>;
using EnginePtr_t = std::shared_ptr<suinfer::engine::IEngine>;
using ServerPtr_t = std::shared_ptr<Server>;
using NetSetupFun = std::function<bool(suinfer::sugraph::INet*, const ServerSettings&)>;
using InputDataPrepareFun = std::function<void(
  BindingHandle* input_bindings, const std::vector<std::size_t>& real_input_size, brPerfServer::Batch* batch,
  std::size_t batch_size, std::size_t compute_unit_num, uint32_t spc_mask, std::size_t device_idx, BesuStream stream,
  const ServerSettings& server_settings, const std::vector<std::size_t>& sample_size)>;
using ResultProcessFun = std::function<void(Batch* batch, void* infer_result_data, std::size_t output_data_size,
                                            std::size_t compute_units, std::size_t single_batch_size)>;
struct InputData {
  ~InputData() = default;
  // sample -> dim adder  [sampleId][dim_idx]
  std::vector<std::vector<void*>> dim_datas;
};

struct Batch {
  Batch() {
    responses.reset(std::make_unique<std::vector<mlperf::QuerySampleResponse>>().release());
    sampleIds.reset(std::make_unique<std::vector<mlperf::QuerySampleIndex>>().release());
  };
  Batch(const Batch& b)
    : responses(std::make_unique<std::vector<mlperf::QuerySampleResponse>>(*b.responses)),
      sampleIds(std::make_unique<std::vector<mlperf::QuerySampleIndex>>(*b.sampleIds)),
      datas(b.datas) {}
  Batch(Batch&& b) : responses(b.responses.release()), sampleIds(b.sampleIds.release()), datas(std::move(b.datas)) {}

  ~Batch() {}

  // We want to move this resource around without copying them.
  std::unique_ptr<std::vector<mlperf::QuerySampleResponse>> responses{nullptr};
  std::unique_ptr<std::vector<mlperf::QuerySampleIndex>> sampleIds{nullptr};

  // Input idx --> std::vector<void*>
  std::vector<InputData> datas;

  BindingHandle* input_bindings{nullptr};
  BindingHandle* output_bindings{nullptr};

  // If there is no packlist, all samples will be packed into batch for one compute unit on average.
  std::vector<std::vector<BertInputDataType>> packlist;
};

struct ServerSettings {
  std::size_t batch_size{0};
  std::size_t gpu_batch_size{256};
  std::size_t num_infer_streams{1};
  std::size_t num_complete_threads{1};
  std::string model_path;
  std::string data_file_suffix;
  bool start_from_device{false};
  bool end_on_device{false};
  bool enable_packing{false};
  bool is_int8{false};
  bool verbose{false};
  bool enable_numa_input{true};
  std::size_t compute_unit{1};
  std::size_t max_packing_size{3};
  std::vector<std::size_t> devices_;

  std::unordered_map<std::string, suinfer::sugraph::Shape> input_shapes;
  suinfer::sugraph::DataType input_data_type{suinfer::sugraph::DataType::Bfloat16};
  suinfer::sugraph::DataLayout input_data_layout{static_cast<suinfer::sugraph::DataLayout>(270)};
  std::vector<std::size_t> input_memory_sizes;

  suinfer::sugraph::DataType output_data_type{suinfer::sugraph::DataType::Bfloat16};
  suinfer::sugraph::DataLayout output_data_layout{static_cast<suinfer::sugraph::DataLayout>(262)};
  std::size_t output_memory_size{0};

  // input --> dims size
  std::vector<std::vector<std::size_t>> dim_sizes;

  std::chrono::microseconds Timeout{10000us};
  std::chrono::microseconds target_latency_ms{10000us};
};

struct ServerParams {
  std::string device_ids;
  std::string scenario;
  std::map<size_t, std::vector<uint32_t>> device_spc_masks;
  std::vector<std::vector<std::vector<std::string>>> EngineNames;
};

// captures execution engine for performing inference
class Device {
public:
  // Use modification function to perform net setting.

  struct EngineInfo {
    EngineInfo(std::unique_ptr<suinfer::engine::IEngine>&& engine_ptr, suinfer::backend::streamHandle_t stream_t,
               std::size_t binding_cnt, short spc_mask = 0)
      : engine(std::move(engine_ptr)), stream(stream_t) {
      bindings = new BindingHandle[binding_cnt];
    }

    ~EngineInfo() {
      delete[] bindings;
      besuStreamDestroy(reinterpret_cast<BesuStream>(stream));
    }

    std::unique_ptr<suinfer::engine::IEngine> engine;
    suinfer::backend::streamHandle_t stream;
    BindingHandle* bindings{nullptr};
    short resource_mask;
    // This is used to get memory offset fast.
    short resource_offset;
  };

public:
  Device(std::size_t device_idx, uint32_t spc_mask, std::size_t samples_to_cache, const ServerSettings& server_setting,
         const NetSetupFun& net_setup, const ResultProcessFun& result_process, const InputDataPrepareFun& data_process)
    : device_idx_(device_idx),
      spc_mask_(spc_mask),
      batch_size_(server_setting.batch_size),
      samples_to_cache_(samples_to_cache),
      model_path_(server_setting.model_path),
      server_settings_(server_setting),
      result_processor_(result_process),
      data_preparer_(data_process) {
    if (0 == spc_mask) {
      spc_mask = GetDeviceSPCMasks(device_idx);
    }
    spc_mask_ = spc_mask;
    compute_unit_num_ = static_cast<size_t>(GetBitNumbersFromMask(spc_mask));
    gpu_batch_size_ = std::min(batch_size_ * compute_unit_num_, server_settings_.gpu_batch_size);

    std::cout << "device_idx_:" << device_idx_ << ", batch_size_:" << batch_size_
              << ", samples_to_cache_:" << samples_to_cache_ << ", compute_unit_num_:" << compute_unit_num_
              << ", gpu_batch_size_:" << gpu_batch_size_ << std::endl;
    _setup(net_setup);
    device_ready_ = true;
  }

  ~Device();

  void infer(Batch* batch) noexcept;
  std::size_t getInputNum() const noexcept { return real_input_size_.size(); }
  std::size_t getOutputNum() const noexcept { return real_output_size_.size(); }

  std::size_t getGPUBatchSize() const noexcept { return std::min(maxOnceProcessNum(), gpu_batch_size_); }

  uint32_t getDeviceIdx() const noexcept { return device_idx_; }

  void reset() noexcept;
  void done() noexcept;

  std::size_t maxOnceProcessNum() const { return batch_size_ * compute_unit_num_; };

  static bool isMemContinuous(const std::vector<std::vector<void*>>& dim_datas, std::size_t dim_idx,
                              const std::vector<std::size_t>& dim_sizes, std::size_t unit_idx, std::size_t sample_num,
                              std::size_t batch_size) noexcept {
    bool is_continuous{true};
    const auto& dim_size = dim_sizes[dim_idx];

    auto* prev_dim_data_block_addr = dim_datas[unit_idx * batch_size][dim_idx];
    for (std::size_t idx = 1; idx < sample_num; ++idx) {
      auto* curr_dim_data_block_addr = dim_datas[unit_idx * batch_size + idx][dim_idx];
      if (curr_dim_data_block_addr != (void*)((uint8_t*)prev_dim_data_block_addr + dim_size)) {
        return !is_continuous;
      }

      prev_dim_data_block_addr = curr_dim_data_block_addr;
    }

    return is_continuous;
  }

private:
  std::string _generateThreadNameWithCardId(const std::string& thread_key);

  // 1st step: fetch ready engine from queue and perfrom run on engine. Then add this batch to this engine
  //           and move this engine into completion queue.
  void _dataPrepare();

  // 2nd step: fetch ready engine from queue and perfrom run on engine. Then add this batch to this engine
  //           and move this engine into completion queue.
  void _performInfer() noexcept;

  // 3rd step: fetch engine from completion queue and sync stream and call QuerySamplesComplete with callback.
  //           then, move this engine into
  void _completion() noexcept;
  void _setup(const NetSetupFun& net_setup) noexcept;
  void _initInputOutputSizes(suinfer::engine::IEngine* engine, suinfer::sugraph::INet* net) noexcept;
  void _allocateDeviceMemory() noexcept;
  void _allocateBuff(BindingHandle* bindings, std::size_t size, bool on_device) noexcept;

private:
  const uint32_t device_idx_{0};
  uint32_t spc_mask_{0};
  std::size_t batch_size_{0};
  std::size_t gpu_batch_size_{0};
  std::size_t samples_to_cache_{0};
  std::size_t compute_unit_num_{1};

  BesuStream data_move_in_stream_;
  BesuStream data_move_out_stream_;

  std::unique_ptr<std::thread> datacombine_thread_;
  // Completion management
  std::unique_ptr<std::thread> complete_thread_;
  std::unique_ptr<std::thread> infer_thread_;
  const int64_t max_device_buff_size_{1024 * 1024 * 1500};
  EngineInfo* free_engine_;

  // input_idx -> data
  QUEUE_TYPE<Batch*> request_queue_;
  QUEUE_TYPE<Batch*> copied_data_queue_;
  QUEUE_TYPE<Batch*> completion_queue_;

  QUEUE_TYPE<std::vector<char*>*> result_buffers_;

  QUEUE_TYPE<BindingHandle*> free_input_binding_buffers_;
  QUEUE_TYPE<BindingHandle*> free_output_binding_buffers_;

  const std::string model_path_;
  std::unique_ptr<suinfer::IBuilder> builder_;
  std::unique_ptr<suinfer::IAppConfigs> configs_;

  // engine needs to be destroyed before net
  std::vector<std::unique_ptr<suinfer::sugraph::INet>> nets_;

  const ServerSettings& server_settings_;

  // input_idx -> sample size
  std::vector<std::size_t> sample_size_;
  std::vector<std::size_t> real_input_size_;
  std::vector<std::size_t> real_output_size_;
  // number of input
  std::size_t input_cnt_{0};
  ResultProcessFun result_processor_;
  InputDataPrepareFun data_preparer_;
  std::atomic_bool device_ready_{false};
  // This is only for convinient to free all buffers.
  std::vector<void*> allocated_buffers_;
  std::vector<std::size_t> engine_mem_offsets_;

  bool stop_{false};
};

// Create buffers and other execution resources.
// Perform queuing, batching, and manage execution resources.
class Server : public mlperf::SystemUnderTest {
public:
  // Query management
  using BatchQueue = QUEUE_TYPE<std::vector<mlperf::QuerySample*>*>;

  Server(std::string name) : name_(name) {}
  ~Server() {
    for (auto& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void AddSampleLibrary(std::size_t device_idx, samplesLoader::SampleLoaderPtr sl) noexcept {
    if (device_idx >= sample_libraries_.size()) {
      sample_libraries_.resize(device_idx + 1);
    }
    sample_libraries_[device_idx] = sl;
  }

  void Setup(const ServerSettings& settings, const ServerParams& params, NetSetupFun net_setup,
             ResultProcessFun result_process, InputDataPrepareFun data_prepare) noexcept;
  void Warmup(double duration) noexcept;
  void Done() noexcept;
  const std::vector<DevicePtr_t>& GetDevices() const noexcept { return vir_devices_; }

  // SUT virtual interface
  const std::string& Name() const override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
  void FlushQueries() override;

private:
  void _packSamplesToList(std::vector<mlperf::QuerySample*>& samples,
                          std::deque<Biren::packer::PackedSequence<BertInputDataType>*>& packed_list,
                          samplesLoader::SampleLoaderPtr qsl, std::size_t batch_size);
#ifdef QUICK_DETECT_PACKED_SIZE
  size_t _detectPackSamplesToListResultSize(std::vector<mlperf::QuerySample*>& samples,
                                            std::deque<Biren::packer::PackedSequence<BertInputDataType>*>& packed_list,
                                            samplesLoader::SampleLoaderPtr qsl, std::size_t batch_size);
#endif // QUICK_DETECT_PACKED_SIZE

  template<typename T>
  void MergeQueryDispatcher(QUEUE_TYPE<T*>* queue) noexcept;
  template<typename T>
  void MergeQueryPackedDispatcher(QUEUE_TYPE<T*>* queue) noexcept;

  inline void BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                         std::vector<mlperf::QuerySample*>::iterator begin,
                         std::vector<mlperf::QuerySample*>::iterator end, std::size_t samples_cnt,
                         Batch& batch) noexcept;
  inline void IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample*>::iterator begin,
                         std::vector<mlperf::QuerySample*>::iterator end) noexcept;

  inline void BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                         std::deque<Biren::packer::PackedSequence<BertInputDataType>*>::iterator begin,
                         std::deque<Biren::packer::PackedSequence<BertInputDataType>*>::iterator end,
                         std::size_t samples_cnt, Batch& batch) noexcept;

  inline void IssueBatch(DevicePtr_t device, std::size_t batch_size,
                         std::deque<Biren::packer::PackedSequence<BertInputDataType>*>::iterator begin,
                         std::deque<Biren::packer::PackedSequence<BertInputDataType>*>::iterator end) noexcept;

  DevicePtr_t GetNextDispatchDevice() noexcept;
  void Reset() noexcept;

private:
  const std::string name_;
  std::size_t device_index_;
  std::vector<std::size_t> device_idxs_;
  std::vector<DevicePtr_t> vir_devices_;
  // multiple devices for one GPU
  std::map<size_t, std::vector<uint32_t>> device_spc_masks_;
  std::size_t valid_device_num_{0};
  std::size_t input_num_;
  std::vector<samplesLoader::SampleLoaderPtr> sample_libraries_;

  ServerSettings server_settings_;
  std::string scenario_;

  BatchQueue request_queue_;
  std::vector<BatchQueue*> work_queues_;

  // Not use shared_ptr for performance considerations.
  std::vector<std::thread> threads_;
};

}; // namespace brPerfServer
