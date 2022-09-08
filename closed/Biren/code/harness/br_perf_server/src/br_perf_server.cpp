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

#include <algorithm>
#include <fstream>
#include <glog/logging.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "br_perf_server.hpp"
#include "loadgen.h"
#include "query_sample_library.h"

#include "timer.h"

namespace Biren {
namespace packer {
PackList PackUsingSPFHP(const std::vector<int>& histogram, int max_sequence_length, int max_sequences_per_pack);
}
} // namespace Biren

std::chrono::nanoseconds t_total;
uint64_t g_samples_size = 0;

namespace brPerfServer {
using namespace std::chrono_literals;
using namespace suinfer;
using namespace Biren::packer;

static constexpr uint64_t AddrAlignment = 4096;

using HostBuffer = QUEUE_TYPE<std::vector<char*>*>;
using BindingBuffer = QUEUE_TYPE<BindingHandle*>;

Device::~Device() {
  auto wait_thread_end = [](const std::unique_ptr<std::thread>& t) {
    if (!t) {
      return;
    }
    if (t->joinable()) {
      t->join();
    }
  };

  wait_thread_end(datacombine_thread_);
  wait_thread_end(infer_thread_);
  wait_thread_end(complete_thread_);

  besuStreamDestroy(data_move_in_stream_);
  besuStreamDestroy(data_move_out_stream_);

  for (auto buff : allocated_buffers_) {
    besuFree(buff);
  }
}

//----------------
// Device
//----------------
void Device::infer(Batch* batch) noexcept { PUSH_QUEUE(request_queue_, batch); }

void Device::reset() noexcept {
  nets_.clear();
  configs_.reset();
  builder_.reset();
}

void Device::_initInputOutputSizes(suinfer::engine::IEngine* engine, suinfer::sugraph::INet* net) noexcept {
  std::unique_ptr<suinfer::engine::IInferDataSet> data_set(engine->allocInferDataSet());

  real_input_size_.clear();
  real_output_size_.clear();
  std::size_t binding_cnt = data_set->bindingCount();
  for (std::size_t data_idx = 0; data_idx < binding_cnt; ++data_idx) {
    auto infer_data = data_set->getInferData(data_idx);
    if (infer_data->is_input) {
      if (infer_data->size == 0) {
        real_input_size_.push_back(server_settings_.input_memory_sizes[input_cnt_]);
      } else {
        real_input_size_.push_back(infer_data->size);
      }
      ++input_cnt_;
    } else {
      if (server_settings_.end_on_device) {
        // Get the memory size aligned by engine.
        std::size_t mem_size = ALIGN(infer_data->size, AddrAlignment);

        auto out_tensor = net->getOutputTensor(0);
        sugraph::TensorMemory tm = out_tensor->getTensorMemory();
        tm.num_regions = compute_unit_num_;
        tm.size_per_region = mem_size;
        tm.type_idx = 3;
        out_tensor->setTensorMemory(tm);
      }
      real_output_size_.push_back(infer_data->size);
    }
  }

  sample_size_.clear();
  for (std::size_t input_idx = 0; input_idx < input_cnt_; ++input_idx) {
    std::size_t dim_cnt = server_settings_.dim_sizes[input_idx].size();
    sample_size_.push_back(
      std::accumulate(server_settings_.dim_sizes[input_idx].begin(), server_settings_.dim_sizes[input_idx].end(), 0));
  }
}

void Device::_allocateBuff(BindingHandle* bindings, std::size_t size, bool on_device) noexcept {
  void* buffer_ptr = nullptr;
  std::size_t out_size{0};
  BesuStatus rt = BesuStatus::BESU_STATUS_SUCCESS;
  if (on_device) {
    rt = besuNumaMallocDevice(&buffer_ptr, &out_size, compute_unit_num_, size, BESU_MEM_ARCH_TYPE_NUMA);
    assert(rt == BesuStatus::BESU_STATUS_SUCCESS);
    rt = besuMemset(buffer_ptr, 0, out_size * compute_unit_num_);
    assert(rt == BesuStatus::BESU_STATUS_SUCCESS);
  } else {
    rt = besuMallocHost(&buffer_ptr, size * compute_unit_num_);
    assert(rt == BesuStatus::BESU_STATUS_SUCCESS);
    rt = besuMemset(buffer_ptr, 0, size * compute_unit_num_);
    assert(rt == BesuStatus::BESU_STATUS_SUCCESS);
  }
  allocated_buffers_.push_back(buffer_ptr);
  *bindings = reinterpret_cast<BindingHandle>(buffer_ptr);
}

void Device::_allocateDeviceMemory() noexcept {
  std::size_t input_size = std::accumulate(real_input_size_.begin(), real_input_size_.end(), 0);
  std::size_t output_size = std::accumulate(real_output_size_.begin(), real_output_size_.end(), 0);
  std::size_t data_size_per_round = (input_size + output_size) * compute_unit_num_;
  std::size_t cachable_round_num = std::floor(((double)max_device_buff_size_) / data_size_per_round);
  std::size_t required_cache_round = std::floor(((double)samples_to_cache_) / (batch_size_ * compute_unit_num_));
  std::size_t round_num_to_cache = std::min(cachable_round_num, required_cache_round);

  std::cout << "input_size: " << input_size << ", output_size: " << output_size
            << ", data_size_per_round:" << data_size_per_round << std::endl;
  std::cout << "round_num_to_cache: " << round_num_to_cache << ", required_cache_round: " << required_cache_round
            << ", round_num_to_cache:" << round_num_to_cache << std::endl;

  // Tuple of total buffer of one input and data size of single batch of this input.
  std::size_t idx = 0;
  for (std::size_t cache_idx = 0; cache_idx < round_num_to_cache; ++cache_idx) {
    BindingHandle* input_bindings = new BindingHandle[real_input_size_.size()];
    idx = 0;
    for (auto size : real_input_size_) {
      _allocateBuff(&input_bindings[idx++], size, server_settings_.start_from_device);
    }

    PUSH_QUEUE(free_input_binding_buffers_, input_bindings);
  }

  for (std::size_t cache_idx = 0; cache_idx < required_cache_round; ++cache_idx) {
    idx = 0;
    BindingHandle* output_bindings = new BindingHandle[real_output_size_.size()];
    for (auto size : real_output_size_) {
      _allocateBuff(&output_bindings[idx++], size, server_settings_.end_on_device);
    }
    PUSH_QUEUE(free_output_binding_buffers_, output_bindings);
  }

  // Allocate host side buffers.
  std::size_t out_idx = 0;
  BindingHandle buff_ptr = nullptr;
  for (auto batch_idx = 0; batch_idx < required_cache_round; ++batch_idx) {
    std::vector<char*>* buff = new std::vector<char*>(real_output_size_.size());
    out_idx = 0;

    for (auto out_size : real_output_size_) {
      _allocateBuff(&buff_ptr, out_size, false);
      buff->operator[](out_idx++) = reinterpret_cast<char*>(buff_ptr);
    }

    PUSH_QUEUE(result_buffers_, buff);
  }
}

void Device::_setup(const NetSetupFun& net_setup) noexcept {
  static constexpr uint32_t k_max_opt_level = 1024;
  static constexpr suinfer::OptimizerVersion verion = suinfer::OptimizerVersion::Codegen_R1;

  if (server_settings_.verbose) {
    std::string file_name = "suinfer_mlperf_log_";
    file_name += _generateThreadNameWithCardId("c");
    file_name += ".txt";
    auto logger = SUINFER_GetDefaultLogger(file_name.c_str());
    builder_.reset(CreateInferBuilder(logger));
  } else {
    builder_.reset(CreateInferBuilder(nullptr));
  }

  besuSetDevice(device_idx_);

  PRINT_CARD_ID(device_idx_);

  besuStreamCreate(&data_move_in_stream_, 0);
  besuStreamCreate(&data_move_out_stream_, 0);

  configs_.reset(builder_->createConfigs());
  auto builder_config = configs_->getBuilderConfig();
  builder_config->setFwdBackendType(FwdBackendType::FWD_BACKEND_KERNELGEN_R1);
  builder_config->setVerbose(false);
  std::cout << "compute_unit_num_: " << compute_unit_num_ << std::endl;
  builder_config->requestProcessUnits(compute_unit_num_, true);
  builder_config->setSpcMask(spc_mask_);

  auto paser = builder_->createParser(*configs_);
  {
    // Parse network from onnx file
    std::cout << "Loading Model " << device_idx_ << std::endl;
    std::unique_ptr<suinfer::parsers::IParser> parser(builder_->createParser(*configs_));
    const auto net = parser->parseFromFile(model_path_.c_str());
    if (!net) {
      std::cout << "Fail to load model " << model_path_ << std::endl;
      // Sorry, we didn't give a graceful shutdown here. This may lead to core dump
      // in multiple devices scenario.
      exit(1);
    }

    net_setup(net, server_settings_);
#ifdef DEBUG
    parser->serializeToFile("trimmed_model.onnx", *net, true);
#endif
    net->applyAllPass(k_max_opt_level, verion);
    nets_.emplace_back(net);

    std::cout << "Building Engine " << device_idx_ << std::endl;
#ifdef DEBUG
    parser->serializeToFile("trimmed_model_1.onnx", *net, true);
#endif
    auto engine = builder_->createEngine(*configs_);
    if (!engine) {
      std::cout << "Create engine fail." << std::endl;
      // Sorry, we didn't give a graceful shutdown here. This may lead to core dump
      // in multiple devices scenario.
      exit(1);
    }

    if (engine->build(*net) != suinferError_t::SUI_STATUS_SUCCESS) {
      std::cout << "Build engine fail." << std::endl;
      // Sorry, we didn't give a graceful shutdown here. This may lead to core dump
      // in multiple devices scenario.
      exit(1);
    }

#ifdef DEBUG
    parser->serializeToFile("after_building_engine.onnx", *net, true);
#endif

    _initInputOutputSizes(engine, net);
    auto stream_idx = engine->getBackend().streamCreate(0);
    free_engine_ = std::make_unique<EngineInfo>(std::unique_ptr<suinfer::engine::IEngine>(engine), stream_idx,
                                                real_input_size_.size() + real_output_size_.size())
                     .release();
  }

  _allocateDeviceMemory();

  datacombine_thread_ = std::make_unique<std::thread>(&Device::_dataPrepare, this);

  infer_thread_ = std::make_unique<std::thread>(&Device::_performInfer, this);

  complete_thread_ = std::make_unique<std::thread>(&Device::_completion, this);
}

void Device::done() noexcept {
  // We can use nullptr to identify the end of work.
  PUSH_QUEUE(request_queue_, std::make_unique<Batch>().release());

  PUSH_QUEUE(copied_data_queue_, std::make_unique<Batch>().release());

  PUSH_QUEUE(completion_queue_, std::make_unique<Batch>().release());
}

std::string Device::_generateThreadNameWithCardId(const std::string& thread_key) {
  std::string t_name = thread_key + std::to_string(device_idx_) + "." + std::to_string(GetSpcNumFromMask(spc_mask_));
  return t_name;
}

void Device::_dataPrepare() {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("dPrep").c_str());

  BindingHandle* input_binding = nullptr;
  Batch* batch = nullptr;
  for (;;) {
    {
      TIME("_dataPrepare get input binding: ", device_idx_);
      input_binding = POP_QUEUE(free_input_binding_buffers_);
    }
    {
      TIME("_dataPrepare get request: ", device_idx_);
      batch = POP_QUEUE(request_queue_);
      if (batch->sampleIds->empty()) {
        break;
      }
    }
    {
      TIME("_dataPrepare data prepare: ", device_idx_);
      data_preparer_(input_binding, real_input_size_, batch, batch_size_, compute_unit_num_, spc_mask_, device_idx_,
                     data_move_in_stream_, server_settings_, sample_size_);
    }

    {
      TIME("_combineMempush batch back:", device_idx_);
      PUSH_QUEUE(copied_data_queue_, batch);
    }
  }
}

void Device::_performInfer() noexcept {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("Infer").c_str());

  Batch* batch = nullptr;
  BindingHandle* output_binding = nullptr;
  besuSetDevice(device_idx_);
  for (;;) {
    {
      TIME("_performInfer get infer data:", device_idx_);
      batch = POP_QUEUE(copied_data_queue_);
      if (!batch || batch->sampleIds->empty()) {
        break;
      }
    }

    auto bindings = free_engine_->bindings;
    std::size_t binding_idx = 0;
    std::size_t idx = 0;
    for (; idx < real_input_size_.size(); ++binding_idx, ++idx) {
      bindings[binding_idx] = batch->input_bindings[idx];
    }

    {
      TIME("_performInfer get output bindings:", device_idx_);
      output_binding = POP_QUEUE(free_output_binding_buffers_);

      for (idx = 0; idx < real_output_size_.size(); ++binding_idx, ++idx) {
        bindings[binding_idx] = output_binding[idx];
      }
    }

    {
      TIME("_performInfer infer:", device_idx_);
      if (free_engine_->engine->runAsync(bindings, free_engine_->stream) != suinferError_t::SUI_STATUS_SUCCESS) {
        std::cout << "Run Async on engine failed. Please refer to log for detailed information." << std::endl;
      }
      batch->output_bindings = output_binding;
      if (besuStreamSynchronize(reinterpret_cast<BesuStream>(free_engine_->stream)) !=
          BesuStatus::BESU_STATUS_SUCCESS) {
        std::cout << "error happened on besuStreamSynchronize. Please refer to log for detailed information."
                  << std::endl;
      }
    }

    {
      TIME("_performInfer push to completion:", device_idx_);
      PUSH_QUEUE(free_input_binding_buffers_, batch->input_bindings);
      PUSH_QUEUE(completion_queue_, batch);
    }
  }
}

void Device::_completion() noexcept {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("complt").c_str());

  Batch* batch = nullptr;
  std::vector<char*>* result_buffer = nullptr;
  for (;;) {
    {
      // If we complete on device. we need a host side buffer to contains the output data.
      // And put the device buffer back to queue for next use quickly.
      if (server_settings_.end_on_device) {
        TIME("_completion get result buff from queue: ", device_idx_);
        result_buffer = POP_QUEUE(result_buffers_);
        if (!result_buffer) {
          break;
        }
      }
    }

    {
      TIME("_completion get result batch from queue:", device_idx_);
      batch = POP_QUEUE(completion_queue_);

      if (!batch || batch->sampleIds->empty()) {
        break;
      }
    }

    {
      if (server_settings_.end_on_device) {
        TIME("_completion copy result:", device_idx_);

        for (std::size_t output_idx = 0; output_idx < real_output_size_.size(); ++output_idx) {
          size_t data_size = real_output_size_[output_idx] * compute_unit_num_;
          size_t offset = 0;
          if (spc_mask_) {
            offset = GetSpcNumFromMask(spc_mask_) * real_output_size_[output_idx];
          }
          besuMemcpyAsync(result_buffer->operator[](output_idx), batch->output_bindings[output_idx] + offset, data_size,
                          data_move_out_stream_);
        }
        if (server_settings_.end_on_device) {
          besuStreamSynchronize(reinterpret_cast<BesuStream>(data_move_out_stream_));
        }
      } else {
        TIME("_completion result process:", device_idx_);
        if (result_processor_) {
          // This function only accept one output data now. Change it more than 1 outputs are required.
          result_processor_(batch, batch->output_bindings[0], real_output_size_[0], compute_unit_num_, batch_size_);
        }
      }
    }
    PUSH_QUEUE(free_output_binding_buffers_, batch->output_bindings);

    {
      if (server_settings_.end_on_device) {
        TIME("_completion result process:", device_idx_);
        if (result_processor_) {
          result_processor_(batch, result_buffer->operator[](0), real_output_size_[0], compute_unit_num_, batch_size_);
        }
      }
    }

    {
      if (server_settings_.end_on_device) {
        TIME("_completion push back result buff:", device_idx_);

        PUSH_QUEUE(result_buffers_, result_buffer);
      }
    }

    delete batch;
  }
}

void Server::Setup(const ServerSettings& settings, const ServerParams& params, NetSetupFun net_setup,
                   ResultProcessFun result_process, InputDataPrepareFun data_prepare) noexcept {
  server_settings_ = settings;
  scenario_ = params.scenario;
  device_spc_masks_ = params.device_spc_masks;
  device_idxs_ = server_settings_.devices_;

  Reset();

  // We assume there will be no device failures during running.
  std::vector<std::thread> device_creators;
  valid_device_num_ = device_idxs_.size();

  auto total_vir_devices = 0;
  size_t max_size_per_one_card = 0;
  for (auto& mask_vec : device_spc_masks_) {
    total_vir_devices += mask_vec.second.size();
    max_size_per_one_card = std::max(max_size_per_one_card, mask_vec.second.size());
  }
  // std::cout << "total_vir_devices: " << total_vir_devices << std::endl;
  std::map<size_t, std::vector<DevicePtr_t>> tmp_vir_devices{};
  std::mutex tmp_vir_devices_mutex;

  vir_devices_.reserve(total_vir_devices + 1);
  device_creators.reserve(total_vir_devices + 1);
  for (auto device_idx : device_idxs_) {
    std::cout << "server_setting gpu_batch_size: " << server_settings_.gpu_batch_size
              << ", batch_size:" << server_settings_.batch_size << std::endl;
    std::cout << "device_idx: " << device_idx << std::endl;
    tmp_vir_devices[device_idx] = {};
    tmp_vir_devices[device_idx].reserve(max_size_per_one_card);
    auto device_spc_masks = device_spc_masks_.find(device_idx);
    if (device_spc_masks != device_spc_masks_.end()) {
      auto& spc_masks = device_spc_masks_[device_idx];
      for (uint32_t spc_mask : spc_masks) {
        std::cout << "device_idx: " << device_idx << ", spc_mask: " << std::hex << spc_mask << std::dec << std::endl;
        auto creator = [this, spc_mask, &net_setup, &result_process, &data_prepare, &tmp_vir_devices,
                        &tmp_vir_devices_mutex](size_t device_idx) {
          auto dev_ptr =
            std::make_shared<Device>(device_idx, spc_mask, sample_libraries_[device_idxs_[0]]->PerformanceSampleCount(),
                                     server_settings_, net_setup, result_process, data_prepare);
          std::unique_lock<std::mutex> l(tmp_vir_devices_mutex);
          tmp_vir_devices[device_idx].push_back(dev_ptr);
        };

        device_creators.emplace_back(std::thread(creator, device_idx));
        std::this_thread::sleep_for(2000ms);
      }
    } else {
      // 0 : auto detect max spc numbers
      std::cout << "device_idx: " << device_idx << ", spc_mask: 0 for auto detect" << std::endl;
      uint32_t spc_mask = 0;
      auto creator = [this, spc_mask, &net_setup, &result_process, &data_prepare, &tmp_vir_devices,
                      &tmp_vir_devices_mutex](size_t device_idx) {
        auto dev_ptr =
          std::make_shared<Device>(device_idx, spc_mask, sample_libraries_[device_idxs_[0]]->PerformanceSampleCount(),
                                   server_settings_, net_setup, result_process, data_prepare);
        std::unique_lock<std::mutex> l(tmp_vir_devices_mutex);
        tmp_vir_devices[device_idx].push_back(dev_ptr);
      };

      device_creators.emplace_back(std::thread(creator, device_idx));
      std::this_thread::sleep_for(2000ms);
    }
  }

  for (auto& t : device_creators) {
    if (t.joinable()) {
      t.join();
    }
  }

  for (int i = 0; i < total_vir_devices;) {
    for (auto device_idx : device_idxs_) {
      auto& devices_vec = tmp_vir_devices[device_idx];
      if (!devices_vec.empty()) {
        vir_devices_.push_back(devices_vec.front());
        devices_vec.erase(devices_vec.begin());
        i++;
      }
    }
  }

  // create batchers/dispatcher
  if (server_settings_.enable_packing) {
    threads_.emplace_back(
      std::thread(&Server::MergeQueryPackedDispatcher<std::vector<mlperf::QuerySample*>>, this, &request_queue_));
  } else {
    threads_.emplace_back(
      std::thread(&Server::MergeQueryDispatcher<std::vector<mlperf::QuerySample*>>, this, &request_queue_));
  }
}

void Server::Done() noexcept {
  for (auto& device : vir_devices_) {
    if (device) {
      device->done();
    }
  }

  std::size_t thread_idx = 0;
  for (auto& thread : threads_) {
    std::vector<mlperf::QuerySample*>* empty_samples = new std::vector<mlperf::QuerySample*>();
    empty_samples->push_back(std::make_unique<mlperf::QuerySample>().release());
    PUSH_QUEUE(request_queue_, empty_samples);
    ++thread_idx;
  }

  // join after we insert the dummy sample
  for (auto& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void Server::Reset() noexcept {
  device_index_ = 0;

  for (auto& device : vir_devices_) {
    device->reset();
  }
}

static inline PackedSequenceList<BertInputDataType> PackSamples(
  std::vector<mlperf::QuerySample>::const_iterator samples_begin,
  std::vector<mlperf::QuerySample>::const_iterator samples_end, samplesLoader::SampleLoaderPtr qsl,
  std::size_t max_packing_size) noexcept {
  PackedSequenceList<BertInputDataType> packed_list;
  std::unordered_map<std::size_t, std::vector<mlperf::QuerySample*>> samples_map;
  std::unordered_map<mlperf::QuerySample*, std::size_t> samples_data_len_map;
  std::size_t data_cnt = std::distance(samples_begin, samples_end);
  std::vector<std::size_t> data_lens(data_cnt);
  std::size_t idx = 0;
  std::size_t seq_len = 0;

  for (; samples_begin < samples_end; ++samples_begin) {
    seq_len = qsl->GetSampleLen(1, samples_begin->index);
    data_lens[idx++] = seq_len;
    samples_map[seq_len].push_back(const_cast<mlperf::QuerySample*>(&*samples_begin));
    samples_data_len_map[const_cast<mlperf::QuerySample*>(&*samples_begin)] =
      qsl->GetOriginSampleLen(1, samples_begin->index);
  }

  std::vector<int> histogram = Biren::packer::histogram<std::size_t>(data_lens, BertMaxSeqLength);
  assert(max_packing_size <= BertMaxSeqPerBatch);

  PackList packList = PackUsingSPFHP(histogram, BertMaxSeqLength, max_packing_size);

  // vector of packed: sampleIdxs, [1 * 6]
  packed_list =
    pack<BertInputDataType>(samples_map, packList, samples_data_len_map, BertMaxSeqLength, BertMaxSeqPerBatch);

  return packed_list;
}

static inline PackedSequenceList<BertInputDataType> PackSamples(
  std::vector<mlperf::QuerySample*>::const_iterator samples_begin,
  std::vector<mlperf::QuerySample*>::const_iterator samples_end, samplesLoader::SampleLoaderPtr qsl,
  std::size_t max_packing_size) noexcept {
  PackedSequenceList<BertInputDataType> packed_list;
  std::unordered_map<std::size_t, std::vector<mlperf::QuerySample*>> samples_map;
  std::unordered_map<mlperf::QuerySample*, std::size_t> samples_data_len_map;
  std::map<mlperf::QuerySample*, std::size_t> samples_to_seq;
  std::size_t data_cnt = std::distance(samples_begin, samples_end);
  std::vector<std::size_t> data_lens(data_cnt);
  std::size_t idx = 0;
  std::size_t seq_len = 0;

  for (; samples_begin < samples_end; ++samples_begin) {
    seq_len = qsl->GetSampleLen(1, (*samples_begin)->index);
    data_lens[idx++] = seq_len;
    samples_map[seq_len].push_back(const_cast<mlperf::QuerySample*>(*samples_begin));
    samples_data_len_map[*samples_begin] = qsl->GetOriginSampleLen(1, (*samples_begin)->index);
  }

  std::vector<int> histogram = Biren::packer::histogram<std::size_t>(data_lens, BertMaxSeqLength);
  assert(max_packing_size <= BertMaxSeqPerBatch);

  PackList packList = PackUsingSPFHP(histogram, BertMaxSeqLength, max_packing_size);

  // vector of packed: sampleIdxs, [1 * 6]
  packed_list =
    pack<BertInputDataType>(samples_map, packList, samples_data_len_map, BertMaxSeqLength, BertMaxSeqPerBatch);

  return packed_list;
}

#ifdef QUICK_DETECT_PACKED_SIZE
static inline size_t DetectPackSamplesListSize(std::vector<mlperf::QuerySample*>::const_iterator samples_begin,
                                               std::vector<mlperf::QuerySample*>::const_iterator samples_end,
                                               samplesLoader::SampleLoaderPtr qsl,
                                               std::size_t max_packing_size) noexcept {
  std::size_t data_cnt = std::distance(samples_begin, samples_end);
  std::vector<std::size_t> data_lens(data_cnt);
  std::size_t idx = 0;
  std::size_t seq_len = 0;

  for (; samples_begin < samples_end; ++samples_begin) {
    seq_len = qsl->GetSampleLen(1, (*samples_begin)->index);
    data_lens[idx++] = seq_len;
  }

  std::vector<int> histogram = Biren::packer::histogram<std::size_t>(data_lens, BertMaxSeqLength);
  assert(max_packing_size <= BertMaxSeqPerBatch);

  PackList packList = PackUsingSPFHP(histogram, BertMaxSeqLength, max_packing_size);

  return packList.size();
}
#endif // QUICK_DETECT_PACKED_SIZE

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  {
    TIME("IssueQuery", 1);
    std::vector<mlperf::QuerySample*>* tmp_samples = new std::vector<mlperf::QuerySample*>();
    // bert and RN50 share the same code
    for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
      tmp_samples->push_back((mlperf::QuerySample*)&(*iter));
    }
    PUSH_QUEUE(request_queue_, tmp_samples);
  }
}

DevicePtr_t Server::GetNextDispatchDevice() noexcept {
  auto vir_devices = GetDevices();
  auto dev_index = device_index_;
  device_index_ = (device_index_ + 1) % vir_devices.size();
  return vir_devices[dev_index];
}

template<typename T>
void Server::MergeQueryDispatcher(QUEUE_TYPE<T*>* queue) noexcept {
  pthread_setname_np(pthread_self(), std::string("dispatcher").c_str());

  std::deque<T*> samples;
  auto device = GetNextDispatchDevice();
  std::vector<mlperf::QuerySample*> tmp_samples;
  tmp_samples.reserve(device->getGPUBatchSize());
  bool quit{false};
  auto first_query_arrive_time = std::chrono::steady_clock::now();
  auto timeout = server_settings_.target_latency_ms - std::chrono::microseconds{8000us};
  std::cout << "server_settings_.target_latency_ms:" << server_settings_.target_latency_ms.count() << std::endl;
  std::cout << "timeout: " << timeout.count() << std::endl;

  for (; !quit;) {
    samples.clear();
    {
      // TIME("acquire samples", 99);
      ACQUIRE_QUEUE_TIMEOUT(request_queue_, samples, timeout);
    }

    if (samples.empty() && tmp_samples.empty()) {
      continue;
    }

    // merge samples
    {
      TIME("process samples", 99);
      auto now_time = std::chrono::steady_clock::now();
      // Use a null (0) id to represent the end of samples
      if (tmp_samples.empty()) {
        first_query_arrive_time = now_time;
      }

      auto batch_size = device->getGPUBatchSize();

      for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
        std::vector<mlperf::QuerySample*>* p_sample_vec = *iter;
        // std::cout << "p_sample_vec: " << (void*)p_sample_vec << std::endl;
        for (auto p_sample : *p_sample_vec) {
          // Use a null (0) id to represent the end of samples
          if (!p_sample->id) {
            quit = true;
            break;
          }
          tmp_samples.push_back(p_sample);
          g_samples_size++;
          if (tmp_samples.size() >= batch_size) {
            {
              TIME("issue batch", 99);
              IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
            }
            device = GetNextDispatchDevice();
            batch_size = device->getGPUBatchSize();
            tmp_samples.clear();
            // refresh timer
            first_query_arrive_time = now_time;
          }
        }
        delete p_sample_vec;
      }

      // dispatch when full or timedout
      if ((now_time - first_query_arrive_time >= timeout) || (tmp_samples.size() >= device->getGPUBatchSize())) {
        auto batch_size = std::min(device->getGPUBatchSize(), tmp_samples.size());
        if (batch_size > 0) {
          TIME("issue batch", 99);
          IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
        }
        device = GetNextDispatchDevice();
        tmp_samples.clear();
      }
    }
  }
}

void Server::_packSamplesToList(std::vector<mlperf::QuerySample*>& samples,
                                std::deque<Biren::packer::PackedSequence<BertInputDataType>*>& packed_list,
                                samplesLoader::SampleLoaderPtr qsl, std::size_t batch_size) {
  if (server_settings_.max_packing_size == 1) {
    for (const auto s : samples) {
      packed_list.emplace_back(new PackedSequence<BertInputDataType>(
        {100, std::vector<mlperf::QuerySample*>{const_cast<mlperf::QuerySample*>(s)},
         std::vector<BertInputDataType>{0, (BertInputDataType)qsl->GetSampleLen(1, s->index)}}));
    }
  } else {
    auto tmp_packings = PackSamples(samples.begin(), samples.end(), qsl, server_settings_.max_packing_size);
    packed_list.insert(packed_list.end(), tmp_packings.begin(), tmp_packings.end());
  }
}

#ifdef QUICK_DETECT_PACKED_SIZE
size_t Server::_detectPackSamplesToListResultSize(
  std::vector<mlperf::QuerySample*>& samples,
  std::deque<Biren::packer::PackedSequence<BertInputDataType>*>& packed_list, samplesLoader::SampleLoaderPtr qsl,
  std::size_t batch_size) {
  size_t new_size = packed_list.size();
  if (server_settings_.max_packing_size == 1) {
    new_size += samples.size();
  } else {
    new_size += DetectPackSamplesListSize(samples.begin(), samples.end(), qsl, server_settings_.max_packing_size);
  }
}
return new_size;
}
#endif // QUICK_DETECT_PACKED_SIZE

template<typename T>
void Server::MergeQueryPackedDispatcher(QUEUE_TYPE<T*>* queue) noexcept {
  pthread_setname_np(pthread_self(), std::string("dispatcherP").c_str());

  // T: mlperf::QuerySample*
  std::deque<T*> samples;
  auto device = GetNextDispatchDevice();
  std::vector<mlperf::QuerySample*> tmp_samples;
  tmp_samples.reserve(64);

  std::size_t batch_size = device->getGPUBatchSize();
  std::vector<mlperf::QuerySample*> not_send_samples;
  not_send_samples.reserve(batch_size);

  std::deque<Biren::packer::PackedSequence<BertInputDataType>*> packed_list;
  size_t detect_packed_size{0};
  bool quit{false};
  auto first_query_arrive_time = std::chrono::steady_clock::now();
  auto timeout = server_settings_.target_latency_ms - std::chrono::microseconds{60000us};
  std::cout << "server_settings_.max_packing_size:" << server_settings_.max_packing_size << std::endl;
  std::cout << "server_settings_.target_latency_ms:" << server_settings_.target_latency_ms.count() << std::endl;
  std::cout << "timeout: " << timeout.count() << std::endl;

  while (!quit) {
    samples.clear();

    { ACQUIRE_QUEUE_TIMEOUT(request_queue_, samples, timeout); }

    if (!not_send_samples.empty()) {
      tmp_samples.insert(tmp_samples.end(), not_send_samples.begin(), not_send_samples.end());
      not_send_samples.clear();
    }

    if (samples.empty() && tmp_samples.empty()) {
      continue;
    }
    auto now_time = std::chrono::steady_clock::now();

    // pack samples into list
    auto qsl = sample_libraries_[device->getDeviceIdx()];
    std::size_t potential_min_sample_size = batch_size * 2.4;
    if (server_settings_.max_packing_size == 1) {
      potential_min_sample_size = batch_size;
    }

    if (tmp_samples.empty()) {
      first_query_arrive_time = now_time;
    }
    for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
      std::vector<mlperf::QuerySample*>* p_sample_vec = *iter;
      g_samples_size += p_sample_vec->size();
      if (!((*p_sample_vec)[0])->id) {
        quit = true;
        break;
      }
      tmp_samples.insert(tmp_samples.end(), p_sample_vec->begin(), p_sample_vec->end());
      delete p_sample_vec;
    }

    if ((now_time - first_query_arrive_time < timeout) && (tmp_samples.size() < potential_min_sample_size)) {
      continue;
    }

    packed_list.clear();
#ifdef QUICK_DETECT_PACKED_SIZE
    detect_packed_size = _detectPackSamplesToListResultSize(tmp_samples, packed_list, qsl, batch_size);
#else
    _packSamplesToList(tmp_samples, packed_list, qsl, batch_size);
    detect_packed_size = packed_list.size();
#endif // QUICK_DETECT_PACKED_SIZE
    {
      // backup ending samples to meet just right packed list size
      if (detect_packed_size >= batch_size || tmp_samples.size() >= potential_min_sample_size) {
        // more than one batch?
        if (detect_packed_size / batch_size > 1) {
          std::cout << "Huge detect_packed_size:" << detect_packed_size << ", tmp_samples.size():" << tmp_samples.size()
                    << std::endl;
        }

        // keep not send samples
        if (detect_packed_size > batch_size && timeout.count() > 0) {
          auto it_last = tmp_samples.end();
          auto list_target_size = detect_packed_size - detect_packed_size % batch_size;
          while (true) {
            it_last--;
            not_send_samples.insert(not_send_samples.begin(), it_last, tmp_samples.end());
            tmp_samples.erase(it_last, tmp_samples.end());
            packed_list.clear();
#ifdef QUICK_DETECT_PACKED_SIZE
            auto new_size = _detectPackSamplesToListResultSize(tmp_samples, packed_list, qsl, batch_size);
#else
            _packSamplesToList(tmp_samples, packed_list, qsl, batch_size);
            auto new_size = packed_list.size();
#endif // QUICK_DETECT_PACKED_SIZE
            if (new_size > list_target_size) {
              continue;
            }
#ifdef QUICK_DETECT_PACKED_SIZE
            // real pack
            _packSamplesToList(tmp_samples, packed_list, qsl, batch_size);
#endif // QUICK_DETECT_PACKED_SIZE
            break;
          }
        } else {
#ifdef QUICK_DETECT_PACKED_SIZE
          // real pack
          _packSamplesToList(tmp_samples, packed_list, qsl, batch_size);
#endif // QUICK_DETECT_PACKED_SIZE
        }

        do {
          auto remaining = std::min(batch_size, packed_list.size());
          auto packed_end = packed_list.begin() + remaining;
          {
            TIME("issue batch", 99);
            IssueBatch(device, batch_size, packed_list.begin(), packed_end);
          }
          device = GetNextDispatchDevice();
          batch_size = device->getGPUBatchSize();
          packed_list.erase(packed_list.begin(), packed_end);
        } while (!packed_list.empty());
        tmp_samples.clear();
        // refresh timer
        first_query_arrive_time = now_time;
        detect_packed_size = 0;
      }
    }

    if ((now_time - first_query_arrive_time >= timeout) && (detect_packed_size > 0)) {
#ifdef QUICK_DETECT_PACKED_SIZE
      // real pack
      _packSamplesToList(tmp_samples, packed_list, qsl, batch_size);
#endif
      auto batch_size = std::min(device->getGPUBatchSize(), packed_list.size());
      {
        TIME("issue batch", 99);
        IssueBatch(device, batch_size, packed_list.begin(), packed_list.end());
      }
      device = GetNextDispatchDevice();
      tmp_samples.clear();
      packed_list.clear();
      detect_packed_size = 0;
    }
  }
}

void Server::IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end) noexcept {
  assert(!sample_libraries_.empty());
  std::size_t device_idx = device->getDeviceIdx();
  Batch* batch = new Batch();

  // This can be member viriable which is set in setup.
  std::size_t input_num = device->getInputNum();
  if (input_num == 0) {
    return;
  }

  batch->datas.resize(input_num);
  batch->responses->reserve(batch_size);
  batch->sampleIds->reserve(batch_size);

  for (auto itr = begin; itr < end; ++itr) {
    batch->responses->push_back({(*itr)->id, 0, 0});
    batch->sampleIds->emplace_back((*itr)->index);
  }

  for (std::size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    BuildBatch(sample_libraries_[device_idx], input_idx, device_idx, begin, end, batch_size, *batch);
  }
  device->infer(batch);
}

void Server::BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                        std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end, std::size_t samples_cnt,
                        Batch& batch) noexcept {
  auto& data_ref = batch.datas[input_idx];
  data_ref.dim_datas.resize(samples_cnt);
  auto dim_cnt = server_settings_.dim_sizes[input_idx].size();
  std::size_t sample_idx = 0;
  for (auto itr = begin; itr < end; ++itr, ++sample_idx) {
    std::vector<void*> data_adders = sl->GetSampleAddress((*itr)->index, input_idx, device_idx);

    data_ref.dim_datas[sample_idx].resize(dim_cnt, nullptr);
    for (std::size_t dim_idx = 0; dim_idx < dim_cnt; ++dim_idx) {
      data_ref.dim_datas[sample_idx][dim_idx] = data_adders[dim_idx];
    }
  }
}

void Server::BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                        std::deque<PackedSequence<BertInputDataType>*>::iterator begin,
                        std::deque<PackedSequence<BertInputDataType>*>::iterator end, std::size_t samples_cnt,
                        Batch& batch) noexcept {
  auto& data_ref = batch.datas[input_idx];
  data_ref.dim_datas.resize(samples_cnt);
  auto dim_cnt = server_settings_.dim_sizes[input_idx].size();
  std::size_t sample_idx = 0;

  for (auto pq = begin; pq < end; ++pq) {
    for (auto sample : (*pq)->samples) {
      std::vector<void*> data_adders = sl->GetSampleAddress(sample->index, input_idx, device_idx);

      data_ref.dim_datas[sample_idx].resize(dim_cnt, nullptr);
      for (std::size_t dim_idx = 0; dim_idx < dim_cnt; ++dim_idx) {
        data_ref.dim_datas[sample_idx][dim_idx] = data_adders[dim_idx];
      }
      ++sample_idx;
    }
  }
}

void Server::IssueBatch(DevicePtr_t device, std::size_t batch_size,
                        std::deque<PackedSequence<BertInputDataType>*>::iterator begin,
                        std::deque<PackedSequence<BertInputDataType>*>::iterator end) noexcept {
  assert(!sample_libraries_.empty());
  std::size_t device_idx = device->getDeviceIdx();
  Batch* batch = new Batch();

  // This can be member viriable which is set in setup.
  std::size_t input_num = device->getInputNum();
  if (input_num == 0) {
    return;
  }

  batch->datas.resize(input_num);
  batch->responses->reserve(batch_size * BertMaxSeqPerBatch);
  batch->sampleIds->reserve(batch_size * BertMaxSeqPerBatch);
  std::size_t total_samples = 0;
  for (auto itr = begin; itr < end; ++itr) {
    for (auto sample : (*itr)->samples) {
      batch->responses->push_back({sample->id, 0, 0});
      batch->sampleIds->emplace_back(sample->index);
      ++total_samples;
    }
    batch->packlist.push_back((*itr)->masks);
  }

  for (std::size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    if (input_idx != BertPosMaskIdx) {
      BuildBatch(sample_libraries_[device_idx], input_idx, device_idx, begin, end, total_samples, *batch);
    }
  }
  { device->infer(batch); }
}

void Server::FlushQueries() {
  // Not used.
}

} // namespace brPerfServer
