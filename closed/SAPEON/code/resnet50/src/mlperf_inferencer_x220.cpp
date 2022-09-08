
#include "mlperf_inferencer_x220.hpp"

#include <glob.h>
#include <glog/logging.h>
#include <math.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

std::vector<std::string> Split(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
};

int MlperfInferencer::Init(const std::string kBinaryDir,
                           const std::string kImageDir,
                           const std::string kAnnotationPath,
                           const size_t kTimeoutMs, const int kDeviceId) {
  std::cout << "current_path :" << std::filesystem::current_path() << std::endl;
  if (!std::filesystem::is_directory(kBinaryDir)) {
    LOG(ERROR) << kBinaryDir << " dir not exists";
    return EXIT_FAILURE;
  }

  if (!std::filesystem::is_directory(kImageDir)) {
    LOG(ERROR) << kImageDir << " dir not exists";
    return EXIT_FAILURE;
  }

  if (!std::filesystem::exists(kAnnotationPath)) {
    LOG(ERROR) << kAnnotationPath << " file not exists";
    return EXIT_FAILURE;
  }

  runtime_handle_ = sapeon_runtime::MakeLocalSapeonRuntime(
      sapeon_runtime::SapeonDeviceType::kDeviceX220, kDeviceId);
  if (runtime_handle_ == nullptr) {
    LOG(ERROR) << "MakeLocalSapeonRuntime() failed";
    return EXIT_FAILURE;
  }

  sapeon_runtime::ResultType result;
  result = runtime_handle_->OpenDevice();
  if (result != sapeon_runtime::ResultType::kResultOk) {
    LOG(ERROR) << "OpenDevice() failed";
    return EXIT_FAILURE;
  }

  result = runtime_handle_->SetModel("NULL", kBinaryDir);
  if (result != sapeon_runtime::ResultType::kResultOk) {
    LOG(ERROR) << "SetModel() failed";
    return EXIT_FAILURE;
  }

  std::ifstream openFile(kAnnotationPath.data());
  if (openFile.is_open()) {
    std::string line;
    std::string replace_str = "bin";
    std::string target_str = "JPEG";
    while (getline(openFile, line)) {
      std::vector<std::string> result = Split(line, ' ');
      result[0].replace(result[0].find(target_str), target_str.length(),
                        replace_str);
      std::string filename = kImageDir + '/' + result[0];
      total_input_filenames_.push_back(filename);
    }
  }

  total_sample_count_ = total_input_filenames_.size();
  performance_sample_count_ = total_sample_count_;

  const size_t kMaxBatch = 18;
  issued_query_samples_buffered_ =
      new client_util::BufferedQueue<mlperf::QuerySample>(kTimeoutMs,
                                                          kMaxBatch);
  return EXIT_SUCCESS;
}

void MlperfInferencer::SutIssueQuery(
    const std::vector<mlperf::QuerySample>& samples) {
  for (auto sample : samples) {
    issued_query_samples_buffered_->Push(sample);
  }
}

void MlperfInferencer::SutFlushQueries() {
  auto& work = main_thread_log.AddWork(__FUNCTION__);
  work.Start();
  if (is_verbose_) {
    std::cout << " SutFlushQueries()" << std::endl;
  }
  work.End();
}

void MlperfInferencer::QslLoadSamplesToRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  auto& work = main_thread_log.AddWork(__FUNCTION__);
  work.Start();
  if (is_verbose_) {
    std::cout << "QslLoadSamplesToRam() start" << std::endl;
  }
  image_sequence_ = (int8_t*)malloc(image_size_ * total_sample_count_);
  std::vector<std::string>::iterator iter;
  int sequence_idx = 0;
  for (iter = total_input_filenames_.begin();
       iter != total_input_filenames_.end(); iter++) {
    FILE* fp = fopen((*iter).c_str(), "rb");
    fread(image_sequence_ + sequence_idx * image_size_, sizeof(int8_t),
          image_size_, fp);
    sequence_idx++;
    fclose(fp);
  }
  if (is_verbose_) {
    std::cout << "QslLoadSamplesToRam() end" << std::endl;
  }
  work.End();
}

void MlperfInferencer::QslUnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  if (is_verbose_) {
    std::cout << "QslUnloadSamplesFromRam()" << std::endl;
  }
  free(image_sequence_);
}

void MlperfInferencer::Start() {
  auto& work = main_thread_log.AddWork(__FUNCTION__);
  work.Start();
  auto preprocess_func = [&](const int i) {
    std::stringstream ss;
    ss << "Preprocess Thread - " << i;
    auto& thread_log = sapeon_runtime::trace_log.AddThread(ss.str());
    auto& work_e2e = thread_log.AddWork("E2E");
    auto& work = thread_log.AddWork("Preprocess");

    printf(" preprocess[%d] start\n", i);
    while (is_thread_running_) {
      work_e2e.Start();
      if (preprocess_preprocess2_queue_.Size() < 500) {
        auto buffered_inputs = issued_query_samples_buffered_->Pop();
        if (buffered_inputs.size() > 0) {
          work.Start();
          sapeon_runtime::Tensor tensor = sapeon_runtime::Tensor(
              input_shape_, sapeon_runtime::DataType::DT_SINT8,
              sapeon_runtime::Tensor::Format::NHWC);
          size_t new_request_id = request_id_++;
          {
            std::unique_lock<std::mutex> lk(request_map_m_);
            request_map_[new_request_id] = {};
          }
          tensor.SetId(new_request_id);

          mlperf::QuerySample query_samples;
          for (size_t idx = 0; idx < buffered_inputs.size(); idx++) {
            query_samples = buffered_inputs[idx];
            {
              std::unique_lock<std::mutex> lk(request_map_m_);
              request_map_[new_request_id].push_back(query_samples.id);
            }
            signed char* dt = tensor.data<signed char>() + idx * image_size_;
            memcpy(dt, image_sequence_ + query_samples.index * image_size_,
                   sizeof(signed char) * 224 * 224 * 3);
          }
          sapeon_runtime::LayerIn* inputs = new sapeon_runtime::LayerIn();
          inputs->emplace_back(sapeon_runtime::Layer{
              0, std::make_shared<sapeon_runtime::Tensor>(tensor)});
          preprocess_preprocess2_queue_.Push(inputs);
          work.End();
        }
        work_e2e.End();
      }
    }
    printf(" preprocess[%d] end\n", i);
  };

  auto preprocess2_func = [&](const int i) {
    std::stringstream ss;
    ss << "Input Conversion Thread - " << i;
    auto& thread_log = sapeon_runtime::trace_log.AddThread(ss.str());
    auto& work_e2e = thread_log.AddWork("E2E");
    auto& work = thread_log.AddWork("Input Conversion");
    printf(" preprocess2[%d] start\n", i);
    while (is_thread_running_) {
      work_e2e.Start();
      if (preprocess2_inf_pqueue_.Size() < 500) {
        auto preprocessed_output = preprocess_preprocess2_queue_.Pop();
        if (preprocessed_output != nullptr) {
          work.Start();
          auto context =
              runtime_handle_->CreateInferenceContext(*preprocessed_output);
          if (context == nullptr) {
            LOG(ERROR) << "CreateInferenceContext() failed";
            work.End();
            return;
          }
          delete preprocessed_output;
          preprocess2_inf_pqueue_.Push(std::move(context));
          work.End();
        }
      }
      work_e2e.End();
    }
    printf("preprocess2[%d] end\n", i);
  };

  auto inference_func = [&](const int i) {
    std::stringstream ss;
    ss << "Inference Thread - " << i;
    auto& thread_log = sapeon_runtime::trace_log.AddThread(ss.str());
    auto& work_e2e = thread_log.AddWork("E2E");
    auto& work = thread_log.AddWork("Inference");
    printf(" inference[%d] start\n", i);
    while (is_thread_running_) {
      work_e2e.Start();
      if (inf_postprocess_pqueue_.Size() < 500) {
        auto preprocess2_output = preprocess2_inf_pqueue_.Pop();
        if (preprocess2_output != nullptr) {

          try {
            work.Start();
            sapeon_runtime::ResultType result =
                runtime_handle_->ExecuteGraph(preprocess2_output);
            work.End();
            if (result == sapeon_runtime::ResultType::kResultOk) {
              inf_postprocess_pqueue_.Push(std::move(preprocess2_output));
            }
          } catch (std::exception& e) {
            printf("Error\n");
          }
        }
        work_e2e.End();
      }
    }
    printf(" inference[%d] end\n", i);
  };

  auto postprocess_func = [&](const int i) {
    std::stringstream ss;
    ss << "Output Thread - " << i;
    auto& thread_log = sapeon_runtime::trace_log.AddThread(ss.str());
    auto& work_e2e = thread_log.AddWork("E2E");
    auto& work_output_conversion = thread_log.AddWork("Output Conversion");
    auto& work_post_process = thread_log.AddWork("Post Processing");
    printf(" postprocess[%d] start\n", i);
    while (is_thread_running_) {
      work_e2e.Start();
      auto inf_output = inf_postprocess_pqueue_.Pop();
      if (inf_output == nullptr) {
        work_output_conversion.End();
        work_e2e.End();
        continue;
      }
      work_output_conversion.Start();
      auto outputs = runtime_handle_->GetResultSync(
          inf_output, sapeon_runtime::Tensor::Format::NWHC);
      sapeon_runtime::Tensor result_tensor = *outputs[0].tensor;
      std::vector<mlperf::ResponseId> request;
      {
        std::unique_lock<std::mutex> lk(request_map_m_);
        if (request_map_.find(inf_output->id) == request_map_.end()) {
          while (true) {
            printf("FATAL : This line should not invoked!\n");
          }
        }
        request = request_map_[inf_output->id];
      }
      work_output_conversion.End();
      work_post_process.Start();
      for (size_t i = 0; i < request.size(); i++) {
        int idx = i * result_tensor.C();
        int top1_idx = 0;
        char top1_score = -127;
        for (int j = 0; j < result_tensor.C(); j++) {
          if (result_tensor.at<char>(idx + j) > top1_score) {
            top1_score = result_tensor.at<char>(idx + j);
            top1_idx = j - 1;
          }
        }
        uintptr_t inferenced_result = (uintptr_t)&top1_idx;
        mlperf::ResponseId response_id = request[i];
        struct mlperf::QuerySampleResponse response {
          response_id, inferenced_result, 4
        };
        mlperf::QuerySamplesComplete(&response, 1);
      }
      work_post_process.End();
      work_e2e.End();
    }
    printf("postprocess[%d] end\n", i);
  };

  for (size_t i = 0; i < kNumOfInputPreprocessThread_; i++) {
    preprocess_threads_.emplace_back(std::thread(preprocess_func, i));
  }
  for (size_t i = 0; i < kNumOfLayoutConvThread_; i++) {
    preprocess2_threads_.emplace_back(std::thread(preprocess2_func, i));
  }
  for (size_t i = 0; i < kNumOfInferenceThread_; ++i) {
    inference_threads_.emplace_back(std::thread(inference_func, i));
  }
  for (size_t i = 0; i < kNumOfPostprocessThread_; ++i) {
    postprocess_threads_.emplace_back(std::thread(postprocess_func, i));
  }
  std::cout << "MlperfInferencer.Start() end" << std::endl;
  work.End();
}

void MlperfInferencer::Stop() { is_thread_running_ = false; }

void MlperfInferencer::Join() {
  for (size_t i = 0; i < kNumOfInputPreprocessThread_; i++) {
    preprocess_threads_[i].join();
  }
  for (size_t i = 0; i < kNumOfLayoutConvThread_; i++) {
    preprocess2_threads_[i].join();
  }
  for (size_t i = 0; i < kNumOfInferenceThread_; i++) {
    inference_threads_[i].join();
  }
  for (size_t i = 0; i < kNumOfPostprocessThread_; i++) {
    postprocess_threads_[i].join();
  }
}