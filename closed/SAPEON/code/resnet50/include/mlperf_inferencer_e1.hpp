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

class MlperfInferencerE1 {
 private:
  std::unique_ptr<sapeon_runtime::Runtime> runtime_handle_[2];
  sapeon_runtime::Tensor::Shape input_shape_ = {18, 224, 224, 3};

  std::vector<std::string> total_input_filenames_;
  client_util::BufferedQueue<mlperf::QuerySample>*
      issued_query_samples_buffered_ = nullptr;

  std::vector<std::thread> processing_threads_;

  const size_t kNumOfProcessingThread_ = 6;
  const size_t kMaxBatch = 18;
  const size_t image_size_ = 224 * 224 * 3;

  std::atomic<bool> is_thread_running_ = true;

  int8_t* image_sequence_;

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

  ~MlperfInferencerE1() {
    if (issued_query_samples_buffered_ != nullptr) {
      delete issued_query_samples_buffered_;
    }
  }
};
