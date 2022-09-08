/**
 * @file client_hapi.hpp
 * @author Heecheol Yang (heecheol.yang@sk.com)
 * @brief Header for Runtime High Level API
 * @version 1.1.0
 * @date 2021-10-20
 *
 * @copyright Copyright (c) 2021 SK TELECOM CO., LTD.
 *
 *
 * @defgroup HAPI Sapeon Runtime High-level API
 * The high-level API provides user-friend interface to application.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <tensor.hpp>

/**
 * @brief Namespace for runtime
 *
 */
namespace sapeon_runtime {

/**@{*/
/**
 * @brief Type to enumerate Sapeon H/W models
 *
 */
enum SapeonDeviceType {
  /**
   * @brief X220 Function Simulator
   *
   */
  kDeviceX220FS = 0,
  /**
   * @brief X220 ASIC
   *
   */
  kDeviceX220,
  /**
   * @brief X220 ASIC w/ DMA driver
   *
   */
  kDeviceX220V2,
  /**
   * @brief X330 Function Simulator
   *
   */
  kDeviceX330FS,
  /**
   * @brief The number of devices models.
   *
   */
  kNumOfDevices
};

/**
 * @brief Result status code
 *
 */
enum ResultType {
  /**
   * @brief OK
   *
   */
  kResultOk = 0,
  /**
   * @brief Not OK
   *
   */
  kResultNOk,
};

/**
 * @brief Inference context which contains input, model, and output context.
 *
 */
class InferenceContext {
 public:
  virtual ~InferenceContext() = default;
  uint64_t id = 0;
};

/**
 * @brief Vector of input or output tensors. Note that Layer is represented as
 * <Layer id, Tensor>.
 *
 */
struct Layer {
  Layer(const int id, const std::shared_ptr<Tensor> tensor)
      : id(id), tensor(tensor) {}
  int id;
  std::shared_ptr<Tensor> tensor;
};
typedef std::vector<Layer> LayerIn;
typedef std::vector<Layer> LayerOut;
/**
 * @defgroup Runtime Runtime
 * Runtime is a software layer which abstracts Sapeon Device and provides
 * serveral services to use it.
 * @ingroup HAPI
 * @{
 */

/**
 * @brief Runtime to run a model with a sapeon device
 *
 */
class Runtime {
 public:
  virtual ~Runtime() = default;

  /**
   * @brief Open actual H/W and run scheduler
   *
   */
  virtual ResultType OpenDevice() = 0;
  /**
   * @brief Initialize device context before running inference by analyzing AIXG
   * and cps and metadata in 'binnaryPath'
   *
   * @param graph_path Path to AIXG file
   * @param binary_path Path to directory which contains cps and metadata
   */
  virtual ResultType SetModel(const std::string& graph_path,
                              const std::string& binary_path) = 0;

  /**
   * @brief A helper function to convert input images into a Tensor
   *
   * @note Don't use this function in actual appliation.
   *
   * @param image_path Image path. Can be a form of regular expression.
   * @param in_shape Input shape
   * @param preprocess Preprocesing function
   * @return Tensor
   */
  virtual Tensor PrepareImage(
      const std::string& image_path, const Tensor::Shape& in_shape,
      std::function<Tensor(const Tensor&)> preprocess = nullptr) const = 0;

  /**
   * @brief Create a Inference Context
   *
   * @param input Vector of <Layer ID, input Tensor>
   * @return std::unique_ptr<LayerIn> Newly created inference context.
   */
  virtual std::unique_ptr<InferenceContext> CreateInferenceContext(
      const LayerIn& input) = 0;

  /**
   * @brief Start inference
   *
   * @param context Inference context to execute.
   * @return ResultType Execution result.
   */
  virtual ResultType ExecuteGraph(
      std::unique_ptr<InferenceContext>& context) = 0;

  /**
   * @brief Check whether inference is finished.
   *
   * @param context
   * @return ResultType Whether the inference is finished or not.
   */
  virtual ResultType CheckInferenceDone(
      const std::unique_ptr<InferenceContext>& context) const = 0;
  /**
   * @brief Wait for the inference t ocomplete.
   *
   * @param context
   * @return ResultType Inference has been finished successfully or not.
   */
  virtual ResultType WaitInferenceDone(
      const std::unique_ptr<InferenceContext>& context) const = 0;

  /**
   * @brief Get the result of an inference.
   *
   * @param context Inference context obtained by CreateInferenceContext()
   * @param format Memory layout format
   * @param async if false, this method waits until the inference is finished.
   *  if true, this method may return immediately even though the inference is
   * not finished yet.
   * @return LayerOut Vector of output data. If
   * async == true, this method may return an empty vector if the inference is
   * not finished yet.
   */
  virtual LayerOut GetResult(
      const std::unique_ptr<InferenceContext>& context, Tensor::Format format,
      DataType datatype = DataType::DT_NONE, const bool async = false,
      std::function<void(void*)> custom_op = nullptr) = 0;

  /**
   * @brief GetResult() with async=false
   *
   * @param context
   * @return std::shared_ptr<LayerOut>
   */
  virtual LayerOut GetResultSync(
      const std::unique_ptr<InferenceContext>& context, Tensor::Format format,
      DataType datatype = DataType::DT_NONE,
      std::function<void(void*)> custom_op = nullptr) = 0;
  /**
   * @brief GetResult() with async=true
   *
   * @param context
   * @return std::shared_ptr<LayerOut>
   */
  virtual LayerOut GetResultAsync(
      const std::unique_ptr<InferenceContext>& context, Tensor::Format format,
      DataType datatype = DataType::DT_NONE,
      std::function<void(void*)> custom_op = nullptr) = 0;
  /**
   * @brief Stop scheduler and close actual H/W
   *
   */
  virtual ResultType CloseDevice() = 0;
};

/**
 * @brief Create one local sapeon runtime.
 *
 * @param device_type Device type
 * @param device_id  Device ID
 * @return std::unique_ptr<Runtime> Pointer to created local runtime instance
 */
std::unique_ptr<Runtime> MakeLocalSapeonRuntime(
    const SapeonDeviceType device_type, const uint8_t device_id);

std::unique_ptr<Runtime> MakeSimpleSapeonRuntime(
    const SapeonDeviceType deviceType, const uint8_t deviceID);
/**@}*/
/**@}*/
}  // namespace sapeon_runtime
