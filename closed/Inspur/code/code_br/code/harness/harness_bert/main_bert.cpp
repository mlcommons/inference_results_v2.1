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

#include <glog/logging.h>
#include <cassert>
#include <iostream>
#include <math.h>
#include <memory>
#include <thread>

#include "backend.h"
#include "br_perf_server.hpp"
#include "loadgen.h"
#include "timer.h"
#include "utils.hpp"

// Flags used by harness
#include "harness_flags.hpp"

extern uint64_t g_samples_size;

auto BertDataProcess = [](brPerfServer::BindingHandle* input_binding, const std::vector<std::size_t>& real_input_size,
                          brPerfServer::Batch* batch, std::size_t batch_size, std::size_t compute_unit_num,
                          uint32_t spc_mask, std::size_t device_idx, BesuStream stream,
                          const brPerfServer::ServerSettings& serverSettings,
                          const std::vector<std::size_t>& sample_size) {
  std::size_t input_cnt = real_input_size.size();
  auto& datas = batch->datas;
  auto& pack_list = batch->packlist;
  std::size_t alignment = 0;
  std::size_t unit_idx = 0;
  char* buff_ptr = nullptr;
  std::size_t data_size = 0;
  std::size_t last_offset = 0;
  std::size_t data_idx = 0;
  std::size_t total_data_len = 0;
  bool no_more_data = false;
  for (std::size_t input_idx = 0; input_idx < input_cnt; ++input_idx) {
    auto& input_datas = datas[input_idx];
    std::size_t sample_idx = 0;
    buff_ptr = reinterpret_cast<char*>(input_binding[input_idx]);
    alignment = real_input_size[input_idx];

    for (std::size_t unit_idx = 0; unit_idx < compute_unit_num && unit_idx < pack_list.size(); ++unit_idx) {
      if (LIKELY(input_idx != BertPosMaskIdx)) {
        auto& one_pack = pack_list[unit_idx];
        last_offset = 0;
        data_idx = 0;
        total_data_len = 0;
        for (auto data_len : one_pack) {
          // The even one in packet is the length of data_size [0, data_size - 1).
          if (++data_idx % 2 == 0) {
            data_size = (data_len - last_offset) * sizeof(BertInputDataType);
            besuMemcpy(buff_ptr, input_datas.dim_datas[sample_idx++][0], data_size);
            total_data_len += data_size;
            buff_ptr += data_size;
            last_offset = data_len;
          }
        }
        buff_ptr += (alignment - total_data_len);
      } else {
        // The input of BertPosMaskIdx is a mask which doesn't need to combine.
        data_size = pack_list[unit_idx].size() * sizeof(BertInputDataType);
        besuMemcpy(buff_ptr, pack_list[unit_idx].data(), data_size);
        buff_ptr += alignment;
      }
    }
  }

  batch->input_bindings = input_binding;
};

auto BertNetSetup = [](suinfer::sugraph::INet* net, const brPerfServer::ServerSettings& serverSettings) -> bool {
  static constexpr uint32_t k_max_opt_level = 1024;
  static constexpr suinfer::OptimizerVersion verion = suinfer::OptimizerVersion::Codegen_R1;

  std::unordered_map<std::string, suinfer::sugraph::Shape> input_shapes;
  for (auto idx = 0; idx < net->getInputTensorCnt(); ++idx) {
    auto tensor = net->getInputTensor(idx);

    if (serverSettings.start_from_device) {
      tensor->setMemoryType(suinfer::backend::MemoryType::MEMORY_DEVICE);
      tensor->setDataLayout(serverSettings.input_data_layout);
      tensor->setDataType(serverSettings.input_data_type);

      std::size_t size_per_sample =
        std::accumulate(serverSettings.dim_sizes[idx].begin(), serverSettings.dim_sizes[idx].end(), 0);
      std::size_t size_per_compute_unit = size_per_sample * serverSettings.batch_size;
      if (serverSettings.enable_numa_input) {
        std::size_t mem_size = ALIGN(size_per_compute_unit, suinfer::backend::k_br_alignment);
        const_cast<brPerfServer::ServerSettings&>(serverSettings).input_memory_sizes.push_back(mem_size);
        tensor->setMemorySize(mem_size);
      } else {
        const_cast<brPerfServer::ServerSettings&>(serverSettings).input_memory_sizes.push_back(size_per_compute_unit);
        tensor->setMemorySize(size_per_compute_unit);
      }
    }

    const auto& itr = serverSettings.input_shapes.find(tensor->getName());
    if (itr != serverSettings.input_shapes.end()) {
      auto shape = itr->second;
      // Setup batch size which is the first dim.
      shape.dim_ints[0] = serverSettings.batch_size;
      input_shapes[tensor->getName()] = shape;
    }
  }

  bool res = net->setInputShapes(input_shapes);

  if (!res) {
    return res;
  }

  auto output_tensor = net->getOutputTensor(0);
  auto output_tensor_1 = net->getOutputTensor(1);
  auto output_tensor_0 = net->getOutputTensor(0);

  // Squeeze
  auto src_op_1 = output_tensor_1->getSrcOp();
  auto src_op_0 = output_tensor_0->getSrcOp();

  net->eraseOutputTensor(output_tensor_0);
  net->eraseOutputTensor(output_tensor_1);

  // Input tensor of squeeze
  output_tensor_0 = src_op_0->getInputTensor(0);
  output_tensor_1 = src_op_1->getInputTensor(0);

  // erase squezze
  net->eraseOp(src_op_0);
  net->eraseOp(src_op_1);

  // split
  src_op_0 = output_tensor_0->getSrcOp();

  // erase input tensors of squeeze
  net->eraseTensor(output_tensor_0);
  net->eraseTensor(output_tensor_1);

  // input tensor of split and output tensor of add.
  output_tensor_0 = src_op_0->getInputTensor(0);

  // erase split
  net->eraseOp(src_op_0);

  net->appendOutputTensor(output_tensor_0);

  net->getOutputTensor(0)->setDataType(serverSettings.output_data_type);
  net->getOutputTensor(0)->setDataLayout(serverSettings.output_data_layout);

  if (serverSettings.end_on_device) {
    net->getOutputTensor(0)->setMemoryType(suinfer::backend::MemoryType::MEMORY_DEVICE);
  }
  return true;
};

auto BertResponseCallback = [](brPerfServer::Batch* batch, void* infer_result_data, std::size_t output_data_size,
                               std::size_t compute_units, std::size_t) {
  static size_t count = 0;

  auto& responses = batch->responses;
  auto start_pos = &(*responses)[0];
  auto end_pos = &(*responses)[responses->size() - 1];

  std::size_t response_cnt = responses->size();
  count += response_cnt;

  // Parse the index in callback.
  mlperf::QuerySamplesComplete(
    start_pos, responses->size(),
    [&responses, compute_units, batch, infer_result_data, output_data_size](mlperf::QuerySampleResponse* response) {
      std::size_t datasize_per_unit = output_data_size;
      if (batch->packlist.empty()) {
        uintptr_t data_ptr = (uintptr_t)infer_result_data;
        for (std::size_t start = 0; start < compute_units; ++start) {
          response->data = (uintptr_t)infer_result_data;
          response->size = datasize_per_unit;
          data_ptr += datasize_per_unit;
        }
      } else {
        std::size_t unit_idx = 0;
        std::size_t pack_idx = 0;
        std::size_t total_pack_idx = 0;
        std::size_t last_start = 0;
        assert(batch->packlist.size() <= compute_units);
        std::size_t sample_idx = 0;
        bool found = false;
        for (auto pack : batch->packlist) {
          const char* pack_result = (char*)infer_result_data + datasize_per_unit * unit_idx++;
          pack_idx = 0;
          for (auto p : pack) {
            if (++pack_idx % 2 == 0) {
              auto size = (p - last_start) * sizeof(BertOutputDataType) * 2;
              if (response->id == (*responses)[sample_idx].id) {
                response->data = (uintptr_t)pack_result;
                response->size = size;
                pack_result += size;
                ++sample_idx;
                return;
              }
              pack_result += size;
              ++sample_idx;
            } else {
              last_start = p;
            }
          }
        }
      }
    });
};

static void InitTestSettings(mlperf::TestSettings& testSettings) {
  testSettings.scenario = scenarioMap[FLAGS_scenario];
  testSettings.mode = testModeMap[FLAGS_test_mode];

  testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
  testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
  testSettings.single_stream_expected_latency_ns = FLAGS_single_stream_expected_latency_ns;
  testSettings.server_coalesce_queries = true;
  testSettings.performance_sample_count_override =
    std::min(testSettings.performance_sample_count_override, FLAGS_performance_sample_count);
}

static void InitLogSettings(mlperf::LogSettings& logSettings) {
  logSettings.log_output.outdir = FLAGS_logfile_outdir;
  logSettings.log_output.prefix = FLAGS_logfile_prefix;
  logSettings.log_output.suffix = FLAGS_logfile_suffix;
  logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
  logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
  logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
  logSettings.log_mode = logModeMap[FLAGS_log_mode];
  logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
  logSettings.enable_trace = FLAGS_log_enable_trace;
}

static void InitServerSettings(brPerfServer::ServerSettings& serverSettings, const mlperf::TestSettings& testSettings) {
  serverSettings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
  if (FLAGS_scenario == "Server") {
    serverSettings.target_latency_ms = std::chrono::microseconds(testSettings.server_target_latency_ns / 1000);
  }

  serverSettings.num_infer_streams = FLAGS_num_infer_streams;
  serverSettings.num_complete_threads = FLAGS_num_complete_threads;
  serverSettings.model_path = FLAGS_model_path;
  serverSettings.verbose = FLAGS_verbose;
  serverSettings.enable_packing = true;
  serverSettings.max_packing_size = FLAGS_max_packing_size;

  serverSettings.is_int8 = false;
  serverSettings.batch_size = GetEngineBatchSizeForModel("bert");
  serverSettings.gpu_batch_size = FLAGS_gpu_batch_size;
  serverSettings.dim_sizes.clear();
  serverSettings.dim_sizes.push_back({512 * sizeof(BertInputDataType)});
  serverSettings.dim_sizes.push_back({512 * sizeof(BertInputDataType)});
  serverSettings.dim_sizes.push_back({512 * sizeof(BertInputDataType)});
  serverSettings.dim_sizes.push_back({6 * sizeof(BertInputDataType)});
  serverSettings.input_shapes["input_ids"] = suinfer::sugraph::Shape(serverSettings.batch_size, 512);
  serverSettings.input_shapes["segment_ids"] = suinfer::sugraph::Shape(serverSettings.batch_size, 512);
  serverSettings.input_shapes["position_ids"] = suinfer::sugraph::Shape(serverSettings.batch_size, 512);
  serverSettings.input_shapes["input_mask"] = suinfer::sugraph::Shape(serverSettings.batch_size, 128);
  serverSettings.input_data_layout = static_cast<suinfer::sugraph::DataLayout>(263);
  serverSettings.input_data_type = suinfer::sugraph::DataType::Float32;
  serverSettings.output_data_layout = static_cast<suinfer::sugraph::DataLayout>(263);
  serverSettings.output_data_type = suinfer::sugraph::DataType::Float32;
  serverSettings.start_from_device = false;
  serverSettings.end_on_device = false;
  serverSettings.data_file_suffix = ".bin";
}

static void InitSpcMasksSettings(brPerfServer::ServerParams& sut_params, const std::vector<int32_t>& devices_list) {
  auto masks_template = {0xffff}; // 16spc

  for (int32_t device_idx : devices_list) {
    std::cout << "device_idx: " << device_idx << ", ";
    auto dev_spc_masks = GetDeviceSPCMasks(device_idx);
    std::cout << "card spc_masks: " << std::hex << dev_spc_masks << ", ";
    sut_params.device_spc_masks[device_idx].clear();
    for (auto mask : masks_template) {
      uint32_t dev_spc_mask = dev_spc_masks & mask;
      if (dev_spc_mask) {
        sut_params.device_spc_masks[device_idx].push_back(dev_spc_mask);
        std::cout << " " << std::hex << dev_spc_mask;
      }
    }
    std::cout << " , total: " << std::dec << sut_params.device_spc_masks[device_idx].size() << std::endl;
  }
}

/* Helper function to actually perform inference using MLPerf Loadgen */
void inference() {
  // Configure the test settings
  mlperf::TestSettings testSettings;
  InitTestSettings(testSettings);

  // Configure the logging settings
  mlperf::LogSettings logSettings;
  InitLogSettings(logSettings);

  // Instantiate and configure our SUT
  brPerfServer::ServerPtr_t server = std::make_shared<brPerfServer::Server>("BR_Bert");

  brPerfServer::ServerSettings serverSettings;
  InitServerSettings(serverSettings, testSettings);

  brPerfServer::ServerParams sut_params;
  sut_params.device_ids = FLAGS_devices;
  sut_params.scenario = FLAGS_scenario;

  sut_params.device_spc_masks = {};

  // Create loaders
  std::cout << "Creating sample loaders." << std::endl;
  std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
  std::vector<bool> start_from_device(tensor_paths.size(), false);

  const std::size_t padding = 0;

  std::map<std::size_t, std::size_t> device_cardIds;
  GetAllDeviceCardIds(device_cardIds);

  std::vector<samplesLoader::SampleLoaderPtr> loaders;
  std::vector<int32_t> devices_list;
  std::vector<int32_t> cardid_list = GetDevicesIdsFromString(sut_params.device_ids);

  for (auto card_id : cardid_list) {
    std::cout << "card_id: " << card_id << std::endl;
    const auto& itr = device_cardIds.find(card_id);
    if (itr == device_cardIds.end()) {
      assert("device_idx set error.");
    }

    devices_list.push_back(itr->second);
  }

  InitSpcMasksSettings(sut_params, devices_list);
  for (int32_t device_idx : devices_list) {
    auto createLoader = [&]() {
      bool trim_zero_endings = false;
      if (serverSettings.enable_packing) {
        trim_zero_endings = true;
      }
      auto loader = std::make_shared<samplesLoader::AppenedSampleLoader>(
        "AppendedSampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","), serverSettings.data_file_suffix,
        FLAGS_performance_sample_count ? FLAGS_performance_sample_count : FLAGS_gpu_batch_size,
        serverSettings.dim_sizes, trim_zero_endings, start_from_device, device_idx, padding);

      loader->setDataTranformerForInput(BertPosIdsIdx, [](char* data, std::size_t len) {
        BertInputDataType* data_ptr = (BertInputDataType*)data;
        std::size_t max_len = len / sizeof(BertInputDataType);
        for (std::size_t i = 0; *data_ptr != 0 && i < max_len; ++data_ptr) {
          *data_ptr = i++;
        }
      });
      serverSettings.devices_.push_back(device_idx);
      server->AddSampleLibrary(device_idx, loader);
      loaders.emplace_back(loader);
    };
    std::thread th(createLoader);
    th.join();
  }
  std::shared_ptr<mlperf::QuerySampleLibrary> aggregated_loader =
    std::shared_ptr<samplesLoader::UniverseSampleLoader>(new samplesLoader::UniverseSampleLoader(loaders));
  std::cout << "Finished Creating sample loader." << std::endl;

  std::cout << "Setting up SUT." << std::endl;
  server->Setup(serverSettings, sut_params, BertNetSetup, BertResponseCallback, BertDataProcess);
  std::cout << "Finished setting up SUT." << std::endl;

  // Perform the inference testing
  std::cout << "Starting running test." << std::endl;
  mlperf::StartTest(server.get(), aggregated_loader.get(), testSettings, logSettings);
  std::cout << "Finished running test." << std::endl;

  server->Done();

  aggregated_loader.reset();
  server.reset();
}

int main(int argc, char* argv[]) {
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging("SuInfer mlperf");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  inference();

  return 0;
}
