/**
 * @file   tensor.hpp
 * @author Yong Hak Lee (camiyu1@gmail.com)
 * @brief  Header only N-dimensional Tensor Class
 * @date   Jun. 2021
 */

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace sapeon_runtime {

enum class DataType : int {
  DT_SINT4 = 0,
  DT_SINT8 = 1,
  DT_SINT16 = 2,
  DT_SMIX48 = 3,
  DT_UINT4 = 4,
  DT_UINT8 = 5,
  DT_UINT16 = 6,
  DT_UMIX48 = 7,
  DT_FP8 = 8,
  DT_FP16 = 9,
  DT_FP32 = 10,
  DT_END = 11,
  DT_NONE = DT_END,
};

/**
 * @brief N-dimensional Tensor Class
 */

class Tensor {
 public:
  enum class Format : int32_t {
    NWHC = 0,
    NHWC = 1,
    NCHW = 2,
    END = 3,
  };

  class Shape {
   public:
    Shape() : d{} {}
    Shape(int d0, int d1, int d2, int d3) : d{d0, d1, d2, d3} {}
    Shape(int d0, int d1, int d2, int d3, Format format) {
      switch (format) {
        case Format::NWHC:
          n_ = 0, w_ = 1, h_ = 2, c_ = 3;
          d[n_] = d0, d[w_] = d1, d[h_] = d2, d[c_] = d3;
          break;
        case Format::NHWC:
          n_ = 0, h_ = 1, w_ = 2, c_ = 3;
          d[n_] = d0, d[h_] = d1, d[w_] = d2, d[c_] = d3;
          break;
        case Format::NCHW:
          n_ = 0, c_ = 1, h_ = 2, w_ = 3;
          d[n_] = d0, d[c_] = d1, d[h_] = d2, d[w_] = d3;
          break;
        default:
          LOG(ERROR) << "Format NYI, forcibily set to NWHC";
          n_ = 0, w_ = 1, h_ = 2, c_ = 3;
          d[n_] = d0, d[w_] = d1, d[h_] = d2, d[c_] = d3;
      }
    }
    //! FIXME: The following two operator[] will be removed
    int& operator[](int i) { return d[i]; }
    const int& operator[](int i) const { return d[i]; }
    size_t size() const { return kRank; }

    bool operator==(Shape b) const {
      return N() == b.N() && W() == b.W() && H() == b.H() && C() == b.C();
    }
    bool operator!=(Shape b) const {
      return N() != b.N() || W() != b.W() || H() != b.H() || C() != b.C();
    }

    void SetN(int val) { d[n_] = val; }
    int N() const { return d[n_]; }
    int W() const { return d[w_]; }
    int H() const { return d[h_]; }
    int C() const { return d[c_]; }

   private:
    //! kRank is now fixed as 4, i.e, Shape is supposed to be 4D.
    static constexpr int kRank{4};
    //! data d array is always arranged as NWHC,
    //! index variables n_, w_, h_, c_ are changed to point as Tensor::Format
    int d[kRank] = {};
    int n_ = 0;
    int w_ = 1;
    int h_ = 2;
    int c_ = 3;

    friend class Tensor;
  };

  virtual ~Tensor() {
    if (this->data_ != nullptr) {
      std::free(this->data_);
      this->data_ = nullptr;
    }
  }

  /**
   * @brief Construct a new Tensor object
   *
   * @param shape : Tensor shape
   * @param datatype : DT_SINT8, DT_UINT8, DT_FP32, ...
   */
  Tensor(const Shape& shape, DataType datatype = DataType::DT_FP32,
         Format format = Format::NWHC)
      : format_(format), shape_(shape), datatype_(datatype) {
    size_t size = GetTensorSize();
    elembytes_ = getElemBytes(datatype_);
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
  }

  Tensor()
      : format_(Format::NWHC),
        shape_(),
        datatype_(DataType::DT_FP32),
        elembytes_(getElemBytes(datatype_)) {}

  Tensor(const std::vector<float>& ref_vector, const Shape& shape,
         Format format = Format::NWHC)
      : format_(format),
        shape_(shape),
        datatype_(DataType::DT_FP32),
        elembytes_(getElemBytes(datatype_)) {
    int size = GetTensorSize();
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    SetData(ref_vector);
  }

  // when pt_size < tensor_size, zero padded
  // when pt_size >= tensor_size, only valid data used
  Tensor(const u_char* pt_src, const size_t pt_size, const Shape& shape,
         DataType datatype = DataType::DT_FP32, Format format = Format::NWHC)
      : format_(format),
        shape_(shape),
        datatype_(datatype),
        elembytes_(getElemBytes(datatype_)) {
    int size = GetTensorSize();
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    size_t tensor_size = size * elembytes_;
    size_t valid_size = std::min(tensor_size, pt_size);
    std::memcpy(data_, pt_src, valid_size);
    size_t remained_size = std::max(tensor_size - valid_size, 0LU);
    std::memset(data_ + valid_size, 0, remained_size);
  }

  Tensor(const Tensor& other)
      : id_(other.id_),
        format_(other.format_),
        shape_(other.shape_),
        datatype_(other.datatype_),
        elembytes_(other.elembytes_) {
    int size = GetTensorSize();
    data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    std::memcpy(data_, other.data_, size * elembytes_);
  }

  Tensor& SetData(const std::vector<float>& ref_vector) {
    size_t size = std::min(ref_vector.size(), GetTensorSize());
    // Initialize the data with ref_vector
    for (size_t i = 0; i < size; ++i) {
      this->at<float>(i) = ref_vector[i];
    }
    // Initialize the remained data to 0
    for (size_t i = size; i < GetTensorSize(); ++i) {
      this->at<float>(i) = 0;
    }
    return *this;
  }

  template <typename T>
  T& at(int idx) const {
    return *(reinterpret_cast<T*>(data_ + idx * elembytes_));
  }

  template <typename T>
  std::vector<T> ToVector() const {
    int size = GetTensorSize();
    std::vector<T> output(size);

    for (int i = 0; i < size; ++i) {
      output.at(i) = this->at<T>(i);
    }
    return output;
  }

  template <typename T>
  T* data() {
    return (reinterpret_cast<T*>(data_));
  }

  template <typename T>
  T* data() const {
    return (reinterpret_cast<T*>(data_));
  }

  Format GetFormat() const { return format_; }

  size_t GetTensorSize() const {
    int size = 1;
    for (int i = 0; i < Shape::kRank; ++i) {
      size *= shape_.d[i];
    }
    return size;
  }

  size_t GetTensorByteSize() const {
    return GetTensorSize() * getElemBytes(datatype_);
  }

  const Shape& GetShape() const { return shape_; }

  DataType GetDataType() const { return datatype_; }

  std::vector<Tensor> SplitBatch(int newN) const {
    if (newN == 0) {
      LOG(ERROR) << "Cannot split tensor into zero batch";
      return std::vector<Tensor>();
    }
    int split_cnt = (GetBatch() + newN - 1) / newN;
    std::vector<Tensor> tensors;
    for (int bidx = 0; bidx < split_cnt; ++bidx) {
      size_t astride = W() * H() * C() * elembytes_;
      size_t gstride = astride * newN;
      size_t remain_batch =
          std::min(newN, std::max(0, GetBatch() - bidx * newN));
      unsigned begin = gstride * bidx;
      uint8_t* ptr = reinterpret_cast<uint8_t*>(data_) + begin;
      size_t size = astride * remain_batch;
      Shape new_shape = shape_;
      new_shape.SetN(newN);
      tensors.emplace_back(ptr, size, new_shape, datatype_, format_);
    }
    return tensors;
  }

  static Tensor MergeBatch(const std::vector<Tensor>& tensors,
                           unsigned valid_batch = 0) {
    // ===begin of condition check===
    // tensor should have at least one tensor
    if (tensors.size() == 0) {
      LOG(ERROR) << "the number of input tensors MUST be larger than 1";
      return Tensor();
    }
    // tensor should have all same astride and format
    const Tensor& ltensor = tensors.front();
    const size_t lastride = ltensor.GetTensorSize() / ltensor.GetBatch();
    const auto lformat = ltensor.GetFormat();
    const auto ldt = ltensor.GetDataType();
    for (const Tensor& tensor : tensors) {
      size_t astride = tensor.GetTensorSize() / tensor.GetBatch();
      auto format = tensor.GetFormat();
      auto dt = tensor.GetDataType();
      if (astride != lastride || format != lformat || dt != ldt) {
        LOG(ERROR) << "Cannot merge inconsistent tensors";
        return Tensor();
      }
    }
    // ===end of condition check===
    // 1. get final shape
    size_t total_batch = 0;
    for (const Tensor& tensor : tensors) {
      total_batch += tensor.GetBatch();
    }
    auto final_shape = ltensor.GetShape();
    final_shape.SetN(
        (valid_batch == 0)
            ? total_batch
            : std::min(static_cast<size_t>(valid_batch), total_batch));
    size_t vbatch = final_shape.N();
    // 1-2. prepare final_tensor
    Tensor final_tensor(final_shape, ltensor.GetDataType(),
                        ltensor.GetFormat());
    uint8_t* ptr = final_tensor.data<uint8_t>();
    // 2. concat all batches
    for (const Tensor& tensor : tensors) {
      size_t vpartial_batch =
          std::min(static_cast<size_t>(tensor.GetBatch()), vbatch);
      vbatch -= vpartial_batch;

      /* byte size calculation */
      size_t bsize_per_batch = tensor.GetTensorByteSize() / tensor.GetBatch();
      size_t bsize_valid = bsize_per_batch * vpartial_batch;
      std::copy(tensor.data<uint8_t>(), tensor.data<uint8_t>() + bsize_valid,
                ptr);
      ptr += bsize_valid;
    }
    return final_tensor;
  }

  // Operator overloading
  bool operator==(const Tensor& other) const {
    if ((this->shape_ != other.shape_) ||
        (this->datatype_ != other.datatype_) ||
        (this->elembytes_ != other.elembytes_) ||
        (this->format_ != other.format_)) {
      return false;
    }

    return (std::memcmp(this->data_, other.data_,
                        other.GetTensorSize() * other.elembytes_) == 0);
  }

  Tensor& operator=(const Tensor& other) {
    if (this == &other) return *this;
    this->shape_ = other.shape_;
    this->format_ = other.format_;

    int size = other.GetTensorSize();

    this->elembytes_ = other.elembytes_;
    this->datatype_ = other.datatype_;

    if (this->data_ != nullptr) std::free(this->data_);
    this->data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    std::copy(other.data_, other.data_ + size * other.elembytes_, this->data_);
    return *this;
  }

  // NWHC specific function
  int N() const { return shape_.N(); }
  int W() const { return shape_.W(); }
  int H() const { return shape_.H(); }
  int C() const { return shape_.C(); }

  int GetBatch() const { return N(); }
  static inline const Shape kInvalidShapeNWHC = {-1, -1, -1, -1};

  void SetId(const uint64_t id) { id_ = id; }
  uint64_t GetId() const { return id_; }

 private:
  uint64_t id_ = 0;
  Format format_ = Format::NWHC;
  Tensor& createTensor(const Shape& shape, DataType datatype) {
    shape_ = shape;
    datatype_ = datatype;
    elembytes_ = getElemBytes(datatype_);
    int size = GetTensorSize();
    this->data_ = static_cast<u_char*>(std::calloc(size, elembytes_));
    return *this;
  }

  int getElemBytes(DataType datatype) const {
    int elembytes;
    switch (datatype) {
      case DataType::DT_FP8:
      case DataType::DT_SINT8:
      case DataType::DT_UINT8:
        elembytes = 1;
        break;
      case DataType::DT_SINT16:
      case DataType::DT_UINT16:
      case DataType::DT_FP16:
        elembytes = 2;
        break;
      case DataType::DT_FP32:
        elembytes = 4;
        break;
      default:
        elembytes = 1;
    }
    return elembytes;
  }

 protected:
  Shape shape_;
  DataType datatype_;
  int elembytes_;
  u_char* data_ = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const DataType& datatype) {
  switch (datatype) {
    case DataType::DT_SINT8:
      os << "SINT8";
      break;
    case DataType::DT_SINT16:
      os << "SINT16";
      break;
    case DataType::DT_UINT8:
      os << "UINT8";
      break;
    case DataType::DT_UINT16:
      os << "UINT16";
      break;
    case DataType::DT_FP8:
      os << "FP8";
      break;
    case DataType::DT_FP16:
      os << "FP16";
      break;
    case DataType::DT_FP32:
      os << "FP32";
      break;
    default:
      os << "not impleted yet";
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Tensor::Format& fmat) {
  switch (fmat) {
    case Tensor::Format::NWHC:
      os << "NWHC";
      break;
    case Tensor::Format::NCHW:
      os << "NCHW";
      break;
    case Tensor::Format::NHWC:
      os << "NHWC";
      break;
    default:
      os << "not impleted yet";
  }
  return os;
}

}  // namespace sapeon_runtime
