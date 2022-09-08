#include <argparse.hpp>
#include <filesystem>

#include "mlperf_inferencer_e1.hpp"
#include "mlperf_inferencer_x220.hpp"

/// \brief Performance unit tests.
namespace perf_tests {

/// \defgroup LoadgenTestsPerformance Test Coverage: Performance

/// \brief A simple SUT implemenatation that immediately completes
/// issued queries sychronously ASAP.
class SystemUnderTestSapeon : public mlperf::SystemUnderTest {
 private:
  MlperfInferencer* inferencer_;
  std::string name_{"SapeonSUT"};

 public:
  SystemUnderTestSapeon(MlperfInferencer* inferencer) {
    inferencer_ = inferencer;
  }
  ~SystemUnderTestSapeon() override = default;
  const std::string& Name() override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    inferencer_->SutIssueQuery(samples);
  }
  void FlushQueries() override { inferencer_->SutFlushQueries(); }
};

/// \brief A stub implementation of QuerySampleLibrary.
class QuerySampleLibrarySapeon : public mlperf::QuerySampleLibrary {
 private:
  MlperfInferencer* inferencer_;
  std::string name_{"SapeonQSL"};

 public:
  QuerySampleLibrarySapeon(MlperfInferencer* inferencer) {
    inferencer_ = inferencer;
  }
  ~QuerySampleLibrarySapeon() = default;
  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() override {
    return inferencer_->total_sample_count_;
  }
  size_t PerformanceSampleCount() override {
    return inferencer_->performance_sample_count_;
  }
  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    inferencer_->QslLoadSamplesToRam(samples);
  }
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    inferencer_->QslUnloadSamplesFromRam(samples);
  }
};

class SystemUnderTestSapeonE1 : public mlperf::SystemUnderTest {
 private:
  MlperfInferencerE1* inferencer_;
  std::string name_{"SapeonSUT"};

 public:
  SystemUnderTestSapeonE1(MlperfInferencerE1* inferencer) {
    inferencer_ = inferencer;
  }
  ~SystemUnderTestSapeonE1() override = default;
  const std::string& Name() override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    inferencer_->SutIssueQuery(samples);
  }
  void FlushQueries() override { inferencer_->SutFlushQueries(); }
};

/// \brief A stub implementation of QuerySampleLibrary.
class QuerySampleLibrarySapeonE1 : public mlperf::QuerySampleLibrary {
 private:
  MlperfInferencerE1* inferencer_;
  std::string name_{"SapeonQSL"};

 public:
  QuerySampleLibrarySapeonE1(MlperfInferencerE1* inferencer) {
    inferencer_ = inferencer;
  }
  ~QuerySampleLibrarySapeonE1() = default;
  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() override {
    return inferencer_->total_sample_count_;
  }
  size_t PerformanceSampleCount() override {
    return inferencer_->performance_sample_count_;
  }
  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    inferencer_->QslLoadSamplesToRam(samples);
  }
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    inferencer_->QslUnloadSamplesFromRam(samples);
  }
};

}  // namespace perf_tests

argparse::ArgumentParser PrepareParser(int argc, char* argv[]) {
  argparse::ArgumentParser parser("SAPEON MLPerf");
  parser.add_argument("-v", "--verbose")
      .default_value(false)
      .help("verbose mode NOT USED");
  parser.add_argument("-b", "--binary_dir")
      .default_value(std::string("../../data/weight_result"))
      .help("binray dir");
  parser.add_argument("-i", "--image_dir")
      .default_value(std::string("../../data/preprocessed_data"))
      .help("input image data dir");
  parser.add_argument("-l", "--label_annotation")
      .default_value(std::string("../../data/val.txt"))
      .help("label annotation txt file");
  parser.add_argument("-o", "--output_dir")
      .default_value(std::string("../log"))
      .help("MLPerf result output dir")
      .action([](const std::string& value) {
        if (!std::filesystem::is_directory(value)) {
          throw std::runtime_error("-o --output_dir [" + value +
                                   "] dir not exists");
        }
        return value;
      });
  parser.add_argument("--short").default_value(false).implicit_value(true).help(
      "loadgen min duration to 60s");
  parser.add_argument("-c", "--user_config")
      .default_value(std::string("../config/user.conf"))
      .help("user config path");
  parser.add_argument("-mc", "--mlperf_config")
      .default_value(std::string("../config/mlperf.conf"))
      .help("user config path");
  parser.add_argument("--suffix")
      .default_value(std::string(""))
      .help("output file suffix name");
  parser.add_argument("--device")
      .required()
      .help("select device [X220-compact, X220-enterprise]")
      .action([](const std::string& value) {
        static const std::vector<std::string> choices = {"X220-compact", "X220-enterprise"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
          std::cout << "device : " << value << std::endl;
          return value;
        }
        throw std::runtime_error("NOT supported device get:[" + value +
                                 "] Check --device option");
      });
  parser.add_argument("-d", "--device_id")
      .default_value(0)
      .help("sapeon device id")
      .scan<'i', int>();
  parser.add_argument("-s", "--scenario")
      .default_value(std::string("Offline"))
      .help(
          "MLPerf Scenario choice [Offline, Server, SingleStream, MultiStream]")
      .action([](const std::string& value) {
        static const std::vector<std::string> choices = {
            "Offline", "Server", "SingleStream", "MultiStream"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
          std::cout << "scenario : " << value << std::endl;
          return value;
        }
        throw std::runtime_error("NOT supported scenario get:[" + value +
                                 "] Check -s option");
      });
  parser.add_argument("-m", "--mode")
      .default_value(std::string("SubmissionRun"))
      .help(
          "MLPerf mode choice [SubmissionRun, AccuracyOnly, PerformanceOnly, "
          "FindPeakPerformance]")
      .action([](const std::string& value) {
        static const std::vector<std::string> choices = {
            "SubmissionRun", "AccuracyOnly", "PerformanceOnly",
            "FindPeakPerformance"};
        if (std::find(choices.begin(), choices.end(), value) != choices.end()) {
          std::cout << "mode : " << value << std::endl;
          return value;
        }
        throw std::runtime_error("NOT supported mode get:[" + value +
                                 "] CHECK -m option");
      });
  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    std::exit(EXIT_FAILURE);
  }
  return parser;
}

int main(int argc, char* argv[]) {
  auto parser = PrepareParser(argc, argv);
  const std::string kBinaryDir = parser.get<std::string>("--binary_dir");
  const std::string kImageDir = parser.get<std::string>("--image_dir");
  const std::string kLabelAnnotation =
      parser.get<std::string>("--label_annotation");
  const std::string kUserConfig = parser.get<std::string>("--user_config");
  const int kDeviceId = parser.get<int>("--device_id");

  int ret = 0;
  const std::string kModelName = "resnet50";
  const std::string kMlperfConfig = parser.get<std::string>("--mlperf_config");
  const size_t kTimeoutMs = 4;

  const std::string kDevice = parser.get<std::string>("--device");

  if (kDevice == "X220-compact") {
    MlperfInferencer inferencer;
    ret = inferencer.Init(kBinaryDir, kImageDir, kLabelAnnotation, kTimeoutMs,
                          kDeviceId);
    if (ret == EXIT_FAILURE) {
      LOG(ERROR) << "MlperfInferencer.init() failed";
      return EXIT_FAILURE;
    }

    perf_tests::SystemUnderTestSapeon sut(&inferencer);
    perf_tests::QuerySampleLibrarySapeon qsl(&inferencer);

    mlperf::TestSettings test_settings;
    std::string scenario_str = parser.get<std::string>("--scenario");
    if (scenario_str == "Offline") {
      test_settings.scenario = mlperf::TestScenario::Offline;
    } else if (scenario_str == "Server") {
      test_settings.scenario = mlperf::TestScenario::Server;
    } else if (scenario_str == "SingleStream") {
      test_settings.scenario = mlperf::TestScenario::SingleStream;
    } else if (scenario_str == "MultiStream") {
      test_settings.scenario = mlperf::TestScenario::MultiStream;
    }

    std::string mode_str = parser.get<std::string>("--mode");
    if (mode_str == "SubmissionRun") {
      test_settings.mode = mlperf::TestMode::SubmissionRun;
    } else if (mode_str == "AccuracyOnly") {
      test_settings.mode = mlperf::TestMode::AccuracyOnly;
    } else if (mode_str == "PerformanceOnly") {
      test_settings.mode = mlperf::TestMode::PerformanceOnly;
    } else if (mode_str == "FindPeakPerformance") {
      test_settings.mode = mlperf::TestMode::FindPeakPerformance;
    }

    test_settings.FromConfig(kMlperfConfig, kModelName, scenario_str);
    test_settings.FromConfig(kUserConfig, kModelName, scenario_str);

    if (parser.get<bool>("--short")) {
      std::cout << "short mode on" << std::endl;
      test_settings.min_duration_ms = 30000;
    }

    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = parser.get<std::string>("--output_dir");
    log_settings.log_output.prefix = "mlperf_log_";
    log_settings.log_output.suffix = parser.get<std::string>("--suffix");
    log_settings.log_output.prefix_with_datetime = false;
    log_settings.log_output.copy_detail_to_stdout = false;
    log_settings.log_output.copy_summary_to_stdout = true;
    log_settings.log_mode = mlperf::LoggingMode::AsyncPoll;
    log_settings.log_mode_async_poll_interval_ms = 1000;
    log_settings.enable_trace = false;

    inferencer.Start();
    mlperf::StartTest(&sut, &qsl, test_settings, log_settings);
    inferencer.Stop();
    inferencer.Join();
  } else {
    MlperfInferencerE1 inferencer;
    ret = inferencer.Init(kBinaryDir, kImageDir, kLabelAnnotation, kTimeoutMs,
                          kDeviceId);
    if (ret == EXIT_FAILURE) {
      LOG(ERROR) << "MlperfInferencerE1.init() failed";
      return EXIT_FAILURE;
    }

    perf_tests::SystemUnderTestSapeonE1 sut(&inferencer);
    perf_tests::QuerySampleLibrarySapeonE1 qsl(&inferencer);

    mlperf::TestSettings test_settings;
    std::string scenario_str = parser.get<std::string>("--scenario");
    if (scenario_str == "Offline") {
      test_settings.scenario = mlperf::TestScenario::Offline;
    } else if (scenario_str == "Server") {
      test_settings.scenario = mlperf::TestScenario::Server;
    } else if (scenario_str == "SingleStream") {
      test_settings.scenario = mlperf::TestScenario::SingleStream;
    } else if (scenario_str == "MultiStream") {
      test_settings.scenario = mlperf::TestScenario::MultiStream;
    }

    std::string mode_str = parser.get<std::string>("--mode");
    if (mode_str == "SubmissionRun") {
      test_settings.mode = mlperf::TestMode::SubmissionRun;
    } else if (mode_str == "AccuracyOnly") {
      test_settings.mode = mlperf::TestMode::AccuracyOnly;
    } else if (mode_str == "PerformanceOnly") {
      test_settings.mode = mlperf::TestMode::PerformanceOnly;
    } else if (mode_str == "FindPeakPerformance") {
      test_settings.mode = mlperf::TestMode::FindPeakPerformance;
    }

    test_settings.FromConfig(kMlperfConfig, kModelName, scenario_str);
    test_settings.FromConfig(kUserConfig, kModelName, scenario_str);

    if (parser.get<bool>("--short")) {
      std::cout << "short mode on" << std::endl;
      test_settings.min_duration_ms = 30000;
    }

    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = parser.get<std::string>("--output_dir");
    log_settings.log_output.prefix = "mlperf_log_";
    log_settings.log_output.suffix = parser.get<std::string>("--suffix");
    log_settings.log_output.prefix_with_datetime = false;
    log_settings.log_output.copy_detail_to_stdout = false;
    log_settings.log_output.copy_summary_to_stdout = true;
    log_settings.log_mode = mlperf::LoggingMode::AsyncPoll;
    log_settings.log_mode_async_poll_interval_ms = 1000;
    log_settings.enable_trace = false;

    inferencer.Start();
    mlperf::StartTest(&sut, &qsl, test_settings, log_settings);
    inferencer.Stop();
    inferencer.Join();
  }

  return EXIT_SUCCESS;
}