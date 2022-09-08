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
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <thread>

#include "backend.h"
#include "br_perf_server.hpp"
#include "loadgen.h"
#include "timer.h"
#include "utils.hpp"

#include "harness_flags.hpp"

extern uint64_t g_samples_size;

auto RN50DataProcess = [](brPerfServer::BindingHandle* input_binding, const std::vector<std::size_t>& real_input_size,
                          brPerfServer::Batch* batch, std::size_t batch_size, std::size_t compute_unit_num,
                          uint32_t spc_mask, std::size_t device_idx, BesuStream stream,
                          const brPerfServer::ServerSettings& server_settings,
                          const std::vector<std::size_t>& sample_size) {
  {
    static size_t count = 0;
    TIME("combine mem:", device_idx);
    auto& sample_ids = *batch->sampleIds;
    std::size_t sample_cnt = sample_ids.size();
    // sample sizes shall be the smaller that all acceptable input data count.
    assert(sample_cnt <= (batch_size * compute_unit_num));

    std::size_t compute_unit_with_data = ceil((float)sample_cnt / batch_size);
    std::size_t offset = 0;
    std::size_t unit_idx = 0;
    std::size_t sample_idx = 0;
    std::size_t input_cnt = real_input_size.size();
    count += sample_cnt;
    for (std::size_t input_idx = 0; input_idx < input_cnt; ++input_idx) {
      std::size_t unit_data_size = batch_size * sample_size[input_idx];
      auto& dim_sizes = server_settings.dim_sizes[input_idx];
      std::size_t dim_cnt = dim_sizes.size();
      // We need to add a groupid to combine data for multiple spc.
      char* data_ptr = reinterpret_cast<char*>(input_binding[input_idx]);

      // assume spcs are continuous
      size_t one_data_size = 0;
      for (auto dim_idx = 0; dim_idx < dim_cnt; ++dim_idx) {
        one_data_size += batch_size * dim_sizes[dim_idx];
      }
      // To align for NUMA memory.
      if (real_input_size[input_idx] > unit_data_size) {
        one_data_size += real_input_size[input_idx] - unit_data_size;
      }

      offset = GetSpcNumFromMask(spc_mask) * one_data_size;
      unit_idx = 0;
      sample_idx = 0;

      // to update how to find out spc idx(unit_idx) if spcs NOT continuous
      for (; unit_idx < compute_unit_with_data; ++unit_idx) {
        if (sample_cnt < unit_idx * batch_size) {
          break;
        }
        // to update how to calc done samples if spcs NOT continuous
        auto sample_num = std::min(sample_cnt - unit_idx * batch_size, batch_size);
        for (auto dim_idx = 0; dim_idx < dim_cnt; ++dim_idx) {
          auto& dim_datas = batch->datas[input_idx].dim_datas;
          if (brPerfServer::Device::isMemContinuous(dim_datas, dim_idx, dim_sizes, unit_idx, sample_num, batch_size)) {
            // If data is continous in memory, we can use this data address directly.
            std::size_t data_size = batch_size * dim_sizes[dim_idx];
            besuMemcpyAsync(data_ptr + offset, dim_datas[unit_idx * batch_size][dim_idx], data_size, stream);
            offset += data_size;
          } else {
            // We are going to copy data batch by batch because one compute unit will consume one batch data.
            for (sample_idx = 0; sample_idx < sample_num; ++sample_idx) {
              besuMemcpyAsync(data_ptr + offset,
                              batch->datas[input_idx].dim_datas[unit_idx * batch_size + sample_idx][dim_idx],
                              dim_sizes[dim_idx], stream);
              offset += dim_sizes[dim_idx];
            }
          }
        }

        // To align for NUMA memory.
        auto align_size = real_input_size[input_idx] - unit_data_size;
        if (align_size > 0) {
          offset += align_size;
        }
      }
    }

    batch->input_bindings = input_binding;
    if (server_settings.start_from_device) {
      TIME("besuStreamSynchronize combine", device_idx);
      besuStreamSynchronize(stream);
    }
  }
};

auto RN50NetSetup = [](suinfer::sugraph::INet* net, const brPerfServer::ServerSettings& server_settings) -> bool {
  static constexpr uint32_t k_max_opt_level = 1024;
  static constexpr suinfer::OptimizerVersion verion = suinfer::OptimizerVersion::Codegen_R1;
  {
    auto input_tensor = net->getInputTensor(0);
    auto shape = input_tensor->getShape();
    shape.dim_ints[0] = server_settings.batch_size;
    std::unordered_map<std::string, suinfer::sugraph::Shape> input_shapes;
    input_shapes[input_tensor->getName()] = shape;
    net->setInputShapes(input_shapes);

    input_tensor->setDataLayout(static_cast<suinfer::sugraph::DataLayout>(275));
    input_tensor->setDataType(server_settings.input_data_type);

    if (server_settings.start_from_device) {
      input_tensor->setMemoryType(suinfer::backend::MemoryType::MEMORY_DEVICE);
    }
    // We need to make alignment when input data is in numa layout.
    std::size_t size_per_sample =
      std::accumulate(server_settings.dim_sizes[0].begin(), server_settings.dim_sizes[0].end(), 0);
    std::size_t size_per_compute_unit = size_per_sample * server_settings.batch_size;
    if (server_settings.enable_numa_input) {
      input_tensor->setMemorySize(ALIGN(server_settings.input_memory_sizes[0], suinfer::backend::k_br_alignment));
    } else {
      input_tensor->setMemorySize(size_per_sample * server_settings.batch_size);
    }
  }
  {
    net->eraseOutputTensor(net->getOutputTensor(1));
    net->eraseOutputTensor(net->getOutputTensor(0));
    auto tensor_name = "resnet_model/dense/BiasAdd:0";
    auto op = net->getOp("resnet_model/dense/BiasAdd:0_DequantizeLinear");
    op->getMajorOutputTensor()->disconnAllDstOps();
    assert(std::strcmp(op->getMajorOutputTensor()->getName(), tensor_name) == 0);
    net->eraseOp("ArgMax");
    net->eraseOp("softmax_tensor");
    net->appendOutputTensor(op->getMajorOutputTensor());
    auto output_tensor = net->getOutputTensor(0);
    output_tensor->setDataType(server_settings.output_data_type);
    output_tensor->setDataLayout(static_cast<suinfer::sugraph::DataLayout>(262));

    if (server_settings.end_on_device) {
      net->getOutputTensor(0)->setMemoryType(suinfer::backend::MemoryType::MEMORY_DEVICE);
    }
  }

  return true;
};

auto RN50ResponseCallback = [](brPerfServer::Batch* batch, void* infer_result_data, std::size_t output_data_size,
                               std::size_t compute_units, std::size_t single_batch_size) {
  static size_t count = 0;
  auto& responses = batch->responses;
  std::size_t response_cnt = responses->size();
  auto out_data = infer_result_data;
  auto out_single_data_size = output_data_size;
  auto out_data_size = out_single_data_size * compute_units;
  for (std::size_t response_idx = 0; response_idx < response_cnt; ++response_idx) {
    if (response_idx == 0) {
      auto& response = responses->operator[](response_idx);
      response.data = (uintptr_t)out_data;
      response.size = out_data_size;
      break;
    }
  }
  count += response_cnt;

  std::size_t response0_id = (*responses)[0].id;
  // Parse the index in callback.
  mlperf::QuerySamplesComplete(
    &(*responses)[0], responses->size(),
    [response0_id, out_single_data_size, response_cnt, compute_units, single_batch_size, batch, &responses,
     infer_result_data](mlperf::QuerySampleResponse* response) {
      std::vector<std::vector<uint32_t>> output_result =
        brPerfServer::DecodeResult((const char*)infer_result_data, out_single_data_size, compute_units,
                                   static_cast<suinfer::sugraph::DataLayout>(262), single_batch_size);

      for (int32_t compute_unit_idx = 0; compute_unit_idx < compute_units; ++compute_unit_idx) {
        for (int32_t batch_idx = 0; batch_idx < single_batch_size; ++batch_idx) {
          auto curr_idx = compute_unit_idx * single_batch_size + batch_idx;
          if (curr_idx >= response_cnt) {
            break;
          }

          if (response->id == (*responses)[curr_idx].id) {
            uint32_t* result_n_ptr = new uint32_t(output_result[compute_unit_idx][batch_idx] - 1);
            response->data = (uintptr_t)result_n_ptr;
            response->size = sizeof(uint32_t);
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

  serverSettings.batch_size = GetEngineBatchSizeForModel("resnet50");
  serverSettings.gpu_batch_size = FLAGS_gpu_batch_size;
  serverSettings.num_infer_streams = FLAGS_num_infer_streams;
  serverSettings.num_complete_threads = FLAGS_num_complete_threads;
  serverSettings.model_path = FLAGS_model_path;
  serverSettings.verbose = FLAGS_verbose;

  serverSettings.dim_sizes.clear();
  serverSettings.dim_sizes.push_back({196608});
  serverSettings.input_data_type = suinfer::sugraph::DataType::Int8;
  serverSettings.input_memory_sizes.push_back(196608 * 16);
  serverSettings.output_data_type = suinfer::sugraph::DataType::Bfloat16;
  serverSettings.output_memory_size = 64064;
  serverSettings.start_from_device = true;
  serverSettings.end_on_device = true;
}

static void InitSpcMasksSettings(brPerfServer::ServerParams& sut_params, const std::vector<int32_t>& devices_list) {
  auto masks_template = {0xffff};

  if (sut_params.scenario == "Server") {
    masks_template = {0xf000, 0xf00, 0xf0, 0xf};
  }

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

  brPerfServer::ServerPtr_t server = std::make_shared<brPerfServer::Server>("BR_RN50");

  brPerfServer::ServerSettings serverSettings;
  InitServerSettings(serverSettings, testSettings);

  brPerfServer::ServerParams sut_params;
  sut_params.device_ids = FLAGS_devices;
  sut_params.scenario = FLAGS_scenario;

  sut_params.device_spc_masks = {};

  std::cout << "Creating samples loader." << std::endl;
  std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
  std::vector<bool> start_from_device(tensor_paths.size(), serverSettings.start_from_device);
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
    auto constructQsl = [&]() {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      bool trim_zero_endings = false;
      if (serverSettings.enable_packing) {
        trim_zero_endings = true;
      }
      auto oneQsl = std::make_shared<samplesLoader::AppenedSampleLoader>(
        "suPerf_SampleLibrary", FLAGS_map_path, splitString(FLAGS_tensor_path, ","), serverSettings.data_file_suffix,
        FLAGS_performance_sample_count ? FLAGS_performance_sample_count : FLAGS_gpu_batch_size,
        serverSettings.dim_sizes, trim_zero_endings, start_from_device, device_idx, padding);

      serverSettings.devices_.push_back(device_idx);
      server->AddSampleLibrary(device_idx, oneQsl);
      loaders.emplace_back(oneQsl);
    };
    std::thread th(constructQsl);
    th.join();
  }
  std::shared_ptr<mlperf::QuerySampleLibrary> aggregated_loader =
    std::shared_ptr<samplesLoader::UniverseSampleLoader>(new samplesLoader::UniverseSampleLoader(loaders));

  std::cout << "Finished Creating samples loader." << std::endl;

  std::cout << "Setting up SUT." << std::endl;
  server->Setup(serverSettings, sut_params, RN50NetSetup, RN50ResponseCallback,
                RN50DataProcess); // Pass the requested server settings and params to our SUT

  std::cout << "Finished setting up SUT." << std::endl;

  std::cout << "Starting running actual test." << std::endl;
  mlperf::StartTest(server.get(), aggregated_loader.get(), testSettings, logSettings);
  std::cout << "Finished running actual test." << std::endl;

  server->Done();

  aggregated_loader.reset();
  server.reset();
}

int main(int argc, char* argv[]) {
  // Initialize logging
  FLAGS_alsologtostderr = 1;
  ::google::InitGoogleLogging("SuInfer mlperf");
  ::google::ParseCommandLineFlags(&argc, &argv, true);

  // Perform inference
  inference();

  return 0;
}
