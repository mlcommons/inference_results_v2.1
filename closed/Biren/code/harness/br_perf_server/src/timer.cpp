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

#include <cassert>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <dirent.h>
#include <sys/stat.h>

#include "suinfer.h"
#include "timer.h"

namespace brPerfServer {

void PerfTimer::printTime(const std::chrono::high_resolution_clock::time_point& p, char* buff, std::size_t size) {
  static constexpr std::size_t k_day_time_length = 19;
  static constexpr std::size_t k_ms_length = 4;
  static constexpr std::size_t k_us_length = 7;
  assert(size >= k_day_time_length + k_us_length - 1);
  auto fraction = p - std::chrono::time_point_cast<std::chrono::seconds>(p);
  std::time_t t = std::chrono::high_resolution_clock::to_time_t(p);
  std::strftime(buff, k_day_time_length + 1, "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));
  snprintf(buff + k_day_time_length, k_us_length + 1, ".%06d",
           static_cast<int>(fraction / std::chrono_literals::operator""us(1)));
}

static std::vector<uint32_t> SoftmaxShape(const std::vector<uint32_t>& in_shape, int32_t axis) {
  // convert in_data shape from {d0, d1, d2,...dn} to {N,D} matrix
  uint32_t N = 1;
  for (uint32_t i = 0; i < axis; ++i) {
    N *= in_shape[i];
  }

  uint32_t D = 1;
  for (uint32_t i = axis; i < in_shape.size(); ++i) {
    D *= in_shape[i];
  }

  return {N, D};
}

static std::vector<float> Softmax(const std::vector<float>& in_data, const std::vector<uint32_t>& in_shape,
                                  int32_t axis) {
  std::vector<float> out_data(in_data.size());
  auto ND = SoftmaxShape(in_shape, axis);
  auto N = ND[0];
  auto D = ND[1];

  for (uint32_t i = 0; i < N; ++i) {
    std::vector<float> exp_sum_list(D);

    for (uint32_t j = 0; j < D; ++j) {
      exp_sum_list[j] = 0.f;
      for (uint32_t k = 0; k < D; ++k) {
        exp_sum_list[j] += exp(in_data[i * D + k] - in_data[i * D + j]);
      }
    }

    for (uint32_t j = 0; j < D; ++j) {
      out_data[i * D + j] = 1.0f / exp_sum_list[j];
    }
  }

  return out_data;
}

std::vector<uint32_t> DecodeResult(const char* predict, std::size_t predict_size, suinfer::sugraph::DataLayout layout,
                                   std::size_t batch, std::size_t class_num) {
  std::vector<uint32_t> result(batch, 0.f);
  for (std::size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    std::vector<float> softmax_input(class_num);

    if (static_cast<uint32_t>(layout) == 262) {
      // backend::KG_A_NCHW_GEMM 262
      // This is 1x1000(X, Y) predict results in y/32 x/8 x4 y32 x2
      // where X/Y align to 32
      // Therefore the dim is Y32 X4 X4 Y32 X2, and we need to
      // iterate on number at X = batch_idx, Y = 0-999
      // aka. Y/32 X/8 (X mode 8)/2 (Y mode 32) (X mode 2)
      const uint16_t* data = reinterpret_cast<const uint16_t*>(predict);
      for (std::size_t i = 0; i < softmax_input.size(); ++i) {
        std::size_t remain = i;
        std::size_t remain_x = batch_idx;
        std::size_t index = 0;
        index = index * 32 + remain / 32; // expand to Y32
        remain %= 32;
        index = index * 4 + remain_x / 8; // expand to X4
        remain_x %= 8;
        index = index * 4 + remain_x / 2; // expand to 2nd X4
        remain_x %= 2;
        index = index * 32 + remain;  // expand to 2nd Y32
        index = index * 2 + remain_x; // expand to X2

        assert(index < (predict_size / sizeof(data[0])));

        uint32_t x = data[index] << 16;
        softmax_input[i] = *reinterpret_cast<float*>(&x);
      }
    } else {
      assert(false);
    }

    std::vector<float>::iterator biggest = std::max_element(std::begin(softmax_input), std::end(softmax_input));
    result[batch_idx] = std::distance(std::begin(softmax_input), biggest);

#if defined(PRINT_SOFTMAX)
    std::vector<uint32_t> in_shape = {static_cast<uint32_t>(softmax_input.size())};
    auto softmax_out = Softmax(softmax_input, in_shape, 0);
    std::sort(softmax_out.begin(), softmax_out.end(), [](float lhs, float rhs) { return lhs > rhs; });
#endif
  }

  return result;
}

std::vector<std::vector<uint32_t>> DecodeResult(const char* predict, std::size_t predict_size,
                                                std::size_t compute_unit_num, suinfer::sugraph::DataLayout layout,
                                                std::size_t batch, std::size_t class_num) {
  std::vector<std::vector<uint32_t>> results(compute_unit_num);
  for (int32_t compute_unit_idx = 0; compute_unit_idx < compute_unit_num; ++compute_unit_idx) {
    results[compute_unit_idx] =
      DecodeResult(predict + predict_size * compute_unit_idx, predict_size, layout, batch, class_num);
  }
  return results;
}

std::vector<std::string> ParseFileList(const char* arg) {
  std::vector<std::string> file_names;

  struct stat file_stat;
  std::string path(arg);
  if (stat(path.c_str(), &file_stat)) {
    printf("Path %s doesn't exist.\n", path.c_str());
    if (!(file_stat.st_mode & (S_IFDIR | S_IFREG))) {
      printf("Path %s is not dir or regular file.\n", path.c_str());
      return file_names;
    }
  }

  if (file_stat.st_mode & S_IFDIR) {
    DIR* dir = opendir(path.c_str());
    if (dir == NULL) {
      printf("Couldn't open directory %s.\n", path.c_str());
      return file_names;
    }
    struct dirent* dir_ent;
    std::string real_file_name;
    while ((dir_ent = readdir(dir)) != NULL) {
      if (dir_ent->d_type == DT_REG) {
        real_file_name = path + '/';
        real_file_name += dir_ent->d_name;
        file_names.emplace_back(real_file_name);
      }
    }
  } else if (file_stat.st_mode & S_IFREG) {
    std::string real_file_name = path;
    file_names.emplace_back(real_file_name);
  }

  std::sort(file_names.begin(), file_names.end());

  return file_names;
}

} // namespace brPerfServer
