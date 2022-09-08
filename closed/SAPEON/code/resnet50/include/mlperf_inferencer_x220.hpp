#pragma once

#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <trace.hpp>
#include <vector>

#include "client_hapi.hpp"
#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_client_util.hpp"
#include "test_settings.h"

class MlperfInferencer {
 private:
  std::unique_ptr<sapeon_runtime::Runtime> runtime_handle_ = nullptr;
  sapeon_runtime::Tensor::Shape input_shape_ = {18, 224, 224, 3};

  std::vector<std::string> total_input_filenames_;
  client_util::BufferedQueue<mlperf::QuerySample>*
      issued_query_samples_buffered_ = nullptr;

  std::vector<std::thread> preprocess_threads_;
  std::vector<std::thread> preprocess2_threads_;
  std::vector<std::thread> inference_threads_;
  std::vector<std::thread> postprocess_threads_;

  size_t kNumOfInputPreprocessThread_ = 1;
  size_t kNumOfLayoutConvThread_ = 1;
  size_t kNumOfInferenceThread_ = 1;
  size_t kNumOfPostprocessThread_ = 1;

  size_t image_size_ = 224 * 224 * 3;

  client_util::Queue<sapeon_runtime::LayerIn*> preprocess_preprocess2_queue_;
  client_util::PtrQueue<sapeon_runtime::InferenceContext>
      preprocess2_inf_pqueue_;
  client_util::PtrQueue<sapeon_runtime::InferenceContext>
      inf_postprocess_pqueue_;

  std::atomic<size_t> request_id_ = 0;

  std::atomic<bool> is_thread_running_ = true;

  int8_t* image_sequence_;

  std::map<size_t, std::vector<mlperf::ResponseId>> request_map_;
  std::mutex request_map_m_;

 public:
  size_t total_sample_count_;
  size_t performance_sample_count_;
  bool is_verbose_ = false;
  mlperf::TestMode mlperf_mode_;
  sapeon_runtime::Trace::Thread& main_thread_log =
      sapeon_runtime::trace_log.AddThread("main");

  int Init(const std::string kBinaryPath, const std::string kImageDir,
           const std::string kAnnotationPath, const size_t timeout_ms,
           const int kDeviceId);
  void SutIssueQuery(const std::vector<mlperf::QuerySample>& samples);
  void SutFlushQueries();
  void QslLoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples);
  void QslUnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples);
  void Start();
  void Stop();
  void Join();

  ~MlperfInferencer() {
    if (issued_query_samples_buffered_ != nullptr) {
      delete issued_query_samples_buffered_;
    }
  }
};