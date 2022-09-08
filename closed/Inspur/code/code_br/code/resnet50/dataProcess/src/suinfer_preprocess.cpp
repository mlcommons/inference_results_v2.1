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

#include <fstream>
#include <cmath>

#include <sys/stat.h>

#include "suinfer.h"

#include "utils.h"
#include "thread_pool.h"
#include "image_loader.h"

const char* images_bin_path = "../data/golden/inputs";
const std::string output_bin_path = "../data/golden/outputs";

void thread_worker(const std::string& filepath) {
  // std::cout << "thread worker start..." << std::endl;

  suinfer::sugraph::Shape i_shape(1, 3, 224, 224); // NCHW
  suinfer::sugraph::Shape o_shape(1, 229, 229, 3); // NCHW
  constexpr size_t output_count = 4;

  constexpr bool is_int8 = true;
  constexpr int input_size_per_pixel = sizeof(uint8_t);
  constexpr int output_size_per_pixel = sizeof(uint8_t);

  std::vector<float> image_decoded_data;
  suinfer::BitmapLoader image;
  image.read(filepath.c_str());
  image.resizeV2(224, 224);
  image_decoded_data = image.decode<float>(is_int8);

  size_t i_shape_lenght = i_shape.length(); // 1 * 3 * 224 * 224;
  std::vector<uint8_t> chw_data(i_shape_lenght, 0);
  quantifer((float*)image_decoded_data.data(), i_shape_lenght, chw_data.data());

  // left, top --> 3, right, down --> 2
  // left , up, right, down
  suinfer::sugraph::Shape padding(3, 3, 2, 2);
  auto left = padding.dim_ints[0];
  auto up = padding.dim_ints[1];
  auto right = padding.dim_ints[2];
  auto down = padding.dim_ints[3];

  auto N = i_shape.dim_ints[0];
  auto C = i_shape.dim_ints[1];
  auto H = i_shape.dim_ints[2];
  auto W = i_shape.dim_ints[3];

  suinfer::sugraph::Shape padding_shape(N, C, H + up + down, W + left + right);
  size_t padding_size = padding_shape.length();

  std::vector<uint8_t> padding_data(padding_size, 0);
  pad<3, 224, 224, 3, 3, 2, 2>(chw_data.data(), padding_data.data());

  size_t o_shape_length = o_shape.length();
  std::vector<uint8_t> hwc_data(o_shape_length, 0);
  CHW2HWC<229, 229, 3>(padding_data.data(), hwc_data.data());

  auto& output = hwc_data;
  generate_outputs(filepath, output.data(),
                   o_shape_length * output_size_per_pixel, output_bin_path);
}

#include <thread>

void preprocess() {
  std::vector<std::string> filenames = ParseFileList(images_bin_path);
  if (filenames.empty()) {
    printf("No files to preprocess.\n");
    return;
  }

  ThreadPool pool(std::thread::hardware_concurrency() - 1);
  pool.init();

  std::vector<std::future<void>> results;
  for (size_t i = 0; i < filenames.size(); ++i) {
    results.emplace_back(pool.submit(thread_worker, filenames[i]));
  }

  for (auto& result : results) {
    result.get();
  }

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(1000ms);
  pool.shutdown();
  std::cout << "preprocess JPEG to padding#split#foldingx4#U#." << std::endl;
}

int main() {
  preprocess();

  std::cout << "Succeeded to run preprocess sample" << std::endl;

  return 0;
}
