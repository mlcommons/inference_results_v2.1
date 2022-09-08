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

#include "utils.h"

#include <fstream>

#include <dirent.h>
#include <sys/stat.h>

#include "math.h"

void write_file(const std::string& path, char* output, size_t len) {
  std::ofstream out_file(path, std::ofstream::binary);
  out_file.write(output, len);
  out_file.flush();
  out_file.close();
}

void read_file(const std::string& path, char* output, size_t len) {
  std::ifstream ifs(path, std::ios::binary);
  ifs.read(output, len);
  ifs.close();
}

std::vector<std::string> ParseFileList(const char* arg) {
  struct stat file_stat;
  const char* path = arg;
  if (stat(path, &file_stat)) {
    printf("Path %s doesn't exist.\n", path);
    if (!(file_stat.st_mode & (S_IFDIR | S_IFREG))) {
      printf("Path %s is not dir or regular file.\n", path);
      return {};
    }
  }

  std::vector<std::string> filenames;
  if (file_stat.st_mode & S_IFDIR) {
    DIR* dir = opendir(path);
    if (dir == NULL) {
      printf("Couldn't open directory %s.\n", path);
      return filenames;
    }
    struct dirent* dir_ent;
    while ((dir_ent = readdir(dir)) != NULL) {
      if (dir_ent->d_type == DT_REG) {
        std::ostringstream ss;
        ss << path << "/" << dir_ent->d_name;
        filenames.emplace_back(ss.str());
      }
    }
  } else if (file_stat.st_mode & S_IFREG) {
    filenames.emplace_back(path);
  }
  std::sort(filenames.begin(), filenames.end());

  return filenames;
}

std::string get_filename(const std::string& filepath) {
  if (filepath.empty())
    return {};

  size_t found = filepath.find_last_of('/');
  if (found == std::string::npos)
    return {};
  return filepath.substr(found + 1);
}

std::string generate_output(const std::string& filepath,
                            const std::string& outpath,
                            const std::string& prefix = "") {
  std::string filename = get_filename(filepath);

  std::ostringstream os;
  os << outpath << "/";
  if (prefix.empty()) {
    os << filename;
  } else {
    os << prefix << "/" << prefix << "_" << filename << ".bin";
  }

  return std::string{os.str()};
}

void generate_outputs(const std::string& filepath, uint8_t* output, size_t size,
                      const std::string& outpath) {
  std::string merge_filename = generate_output(filepath, outpath);
  write_file(merge_filename, (char*)output, size);

  static size_t cnt = 1;
  std::cout << "done: " << cnt++ << std::endl;
}

uint8_t float32touint8(float* in, float y_scale, int32_t y_zero_point) {
  int32_t i32_in = static_cast<int32_t>(std::round(*in / y_scale));
  i32_in += y_zero_point;
  if (i32_in <= 0)
    return 0;

  uint8_t out = (uint8_t)i32_in;
  return out;
}

void quantifer(float* data, size_t size, uint8_t* output) {
  constexpr float y_scale = 1.07741177f;
  constexpr int32_t y_zero_point = 115;

  for (size_t i = 0; i < size; ++i) {
    *(output + i) = float32touint8(data + i, y_scale, y_zero_point);
  }
}