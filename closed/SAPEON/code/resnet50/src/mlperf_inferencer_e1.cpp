#include "mlperf_inferencer_e1.hpp"

#include <glob.h>
#include <glog/logging.h>
#include <math.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <trace.hpp>

std::vector<std::string> SplitE1(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
};

int MlperfInferencerE1::Init(const std::string kBinaryDir,
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

  for (int i = kDeviceId; i < 2; ++i) {
    std::cout << "Init Runtime " << i << std::endl;
    runtime_handle_[i] = sapeon_runtime::MakeLocalSapeonRuntime(
        sapeon_runtime::SapeonDeviceType::kDeviceX220, i);
    if (runtime_handle_[i] == nullptr) {
      LOG(ERROR) << "MakeLocalSapeonRuntime() failed";
      return EXIT_FAILURE;
    }

    sapeon_runtime::ResultType result;
    result = runtime_handle_[i]->OpenDevice();
    if (result != sapeon_runtime::ResultType::kResultOk) {
      LOG(ERROR) << "OpenDevice() failed";
      return EXIT_FAILURE;
    }

    result = runtime_handle_[i]->SetModel("NULL", kBinaryDir);
    if (result != sapeon_runtime::ResultType::kResultOk) {
      LOG(ERROR) << "SetModel() failed";
      return EXIT_FAILURE;
    }
  }

  std::ifstream openFile(kAnnotationPath.data());
  if (openFile.is_open()) {
    std::string line;
    std::string replace_str = "bin";
    std::string target_str = "JPEG";
    while (getline(openFile, line)) {
      std::vector<std::string> result = SplitE1(line, ' ');
      result[0].replace(result[0].find(target_str), target_str.length(),
                        replace_str);
      std::string filename = kImageDir + '/' + result[0];
      total_input_filenames_.push_back(filename);
    }
  }

  total_sample_count_ = total_input_filenames_.size();
  performance_sample_count_ = total_sample_count_;

  issued_query_samples_buffered_ =
      new client_util::BufferedQueue<mlperf::QuerySample>(kTimeoutMs,
                                                          kMaxBatch);
  return EXIT_SUCCESS;
}

void MlperfInferencerE1::SutIssueQuery(
    const std::vector<mlperf::QuerySample>& samples) {
  for (auto sample : samples) {
    issued_query_samples_buffered_->Push(sample);
  }
}

void MlperfInferencerE1::SutFlushQueries() {
  auto& work = main_thread_log.AddWork(__FUNCTION__);
  work.Start();
  if (is_verbose_) {
    std::cout << "SutFlushQueries()" << std::endl;
  }
  work.End();
}

void MlperfInferencerE1::QslLoadSamplesToRam(
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

void MlperfInferencerE1::QslUnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex>& samples) {
  if (is_verbose_) {
    std::cout << "QslUnloadSamplesFromRam()" << std::endl;
  }
  free(image_sequence_);
}

void MlperfInferencerE1::Start() {
  if (is_verbose_) {
    std::cout << "Start()" << std::endl;
  }

  auto& work = main_thread_log.AddWork(__FUNCTION__);
  work.Start();

  auto processing_func = [&](const int i) {
    std::stringstream ss;
    ss << "Processing Thread - " << i;
    auto& thread_log = sapeon_runtime::trace_log.AddThread(ss.str());
    auto& pre_processing_work = thread_log.AddWork("Pre Processing");
    auto& input_conversion_work = thread_log.AddWork("Input Conversion");
    auto& inference_work = thread_log.AddWork("Inference");
    auto& output_conversion_work = thread_log.AddWork("Output conversion");
    auto& post_processing_work = thread_log.AddWork("Post Processing");

    printf(" processing[%d] start\n", i);
    const int runtime_id = i % 2;
    while (is_thread_running_) {
      auto buffered_inputs = issued_query_samples_buffered_->Pop();
      if (buffered_inputs.size() > 0) {
        // pre
        pre_processing_work.Start();
        sapeon_runtime::Tensor tensor = sapeon_runtime::Tensor(
            input_shape_, sapeon_runtime::DataType::DT_SINT8,
            sapeon_runtime::Tensor::Format::NHWC);

        for (size_t idx = 0; idx < buffered_inputs.size(); ++idx) {
          const mlperf::QuerySample& query_sample = buffered_inputs[idx];
          signed char* dt = tensor.data<signed char>() + idx * image_size_;
          memcpy(dt, image_sequence_ + query_sample.index * image_size_,
                 sizeof(signed char) * image_size_);
        }

        sapeon_runtime::LayerIn inputs;
        inputs.emplace_back(sapeon_runtime::Layer{
            0, std::make_shared<sapeon_runtime::Tensor>(tensor)});
        pre_processing_work.End();

        input_conversion_work.Start();
        auto context =
            runtime_handle_[runtime_id]->CreateInferenceContext(inputs);
        input_conversion_work.End();

        if (context == nullptr) {
          LOG(ERROR) << "CreateInferenceContext() failed";
          return;
        }

        // inference
        try {
          inference_work.Start();
          sapeon_runtime::ResultType result =
              runtime_handle_[runtime_id]->ExecuteGraph(context);
          if (result != sapeon_runtime::ResultType::kResultOk) {
            LOG(ERROR) << "ExecuteGraph() failed";
            return;
          }
        } catch (std::exception& e) {
          printf("Error\n");
        }

        // post
        struct mlperf::QuerySampleResponse responses[kMaxBatch];
        int top1_idx[kMaxBatch];

        output_conversion_work.Start();
        auto outputs = runtime_handle_[runtime_id]->GetResultSync(
            context, sapeon_runtime::Tensor::Format::NWHC);
        sapeon_runtime::Tensor result_tensor = *outputs[0].tensor;
        output_conversion_work.End();

        post_processing_work.Start();
        for (size_t i = 0; i < buffered_inputs.size(); i++) {
          int idx = i * result_tensor.C();
          top1_idx[i] = 0;
          char top1_score = -127;
          char* ptr = result_tensor.data<char>() + idx;
          for (int j = 0; j < result_tensor.C(); j++) {
            if (*(ptr + j) > top1_score) {
              top1_score = *(ptr + j);
              // top1_idx[i] = j;
              top1_idx[i] = j - 1;
            }
          }
          responses[i].id = buffered_inputs[i].id;
          responses[i].data = (uintptr_t)(top1_idx + i);
          responses[i].size = 4;
        }
        mlperf::QuerySamplesComplete(responses, buffered_inputs.size());
        post_processing_work.End();
      }
    }
    printf(" processing[%d] end\n", i);
  };
  for (size_t i = 0; i < kNumOfProcessingThread_; ++i) {
    processing_threads_.emplace_back(std::thread(processing_func, i));
  }
  std::cout << "MlperfInferencerE1.Start() end" << std::endl;
  work.End();
}

void MlperfInferencerE1::Stop() {
  if (is_verbose_) {
    std::cout << "Stop()" << std::endl;
  }
  is_thread_running_ = false;
}

void MlperfInferencerE1::Join() {
  if (is_verbose_) {
    std::cout << "Join()" << std::endl;
  }
  for (size_t i = 0; i < kNumOfProcessingThread_; ++i) {
    processing_threads_[i].join();
  }
}
