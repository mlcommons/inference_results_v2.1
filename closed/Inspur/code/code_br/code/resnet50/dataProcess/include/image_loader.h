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
#pragma once

#include <stdexcept>
#include <cassert>
#include <FreeImage.h>
#include <vector>

namespace suinfer {

static constexpr std::size_t k_channel = 3;

class BitmapLoader {
public:
  BitmapLoader() = default;
  virtual ~BitmapLoader();

  inline void read(const char* fname);
  inline void resize(std::size_t h, std::size_t w);
  inline void resizeV2(std::size_t h, std::size_t w);
  inline const uint8_t* scanLine(std::size_t h);
  inline bool isGray() { return is_gray_; }
  template<class _V>
  inline std::vector<_V> decode(bool is_int8 = false);
  template<class _V>
  inline std::vector<_V> decodeOrigin();
  inline std::size_t getWidth() const { return height_; }
  inline std::size_t getHeight() const { return width_; }
  inline uint8_t* getRawPtr() const {
    return bitmap_ ? (uint8_t*)FreeImage_GetBits(bitmap_) : nullptr;
  }

private:
  FIBITMAP* bitmap_ = nullptr;
  std::size_t height_ = 0;
  std::size_t width_ = 0;
  bool is_gray_ = false;
};

BitmapLoader::~BitmapLoader() {
  if (bitmap_) {
    FreeImage_Unload(bitmap_);
  }
}

void BitmapLoader::read(const char* fname) {
  // declare a host image object for an 8-bit grayscale image
  // load gray-scale image from disk
  // set your own bitmaploader error handler
  // FreeImage_setoutputmessage(bitmaploadererrorhandler);
  FREE_IMAGE_FORMAT eformat = FreeImage_GetFileType(fname);

  // no signature? try to guess the file format from the file extension
  if (eformat == FIF_UNKNOWN) {
    eformat = FreeImage_GetFIFFromFilename(fname);
  }

  if (eformat == FIF_UNKNOWN) {
    throw std::runtime_error("unknown image format");
  }

  // check that the plugin has reading capabilities ...
  if (FreeImage_FIFSupportsReading(eformat)) {
    bitmap_ = FreeImage_Load(eformat, fname);
  }

  if (!bitmap_) {
    throw std::runtime_error("error reading image");
  }

  height_ = FreeImage_GetHeight(bitmap_);
  width_ = FreeImage_GetWidth(bitmap_);
  is_gray_ = (FreeImage_GetBPP(bitmap_) == 8);
}

void BitmapLoader::resize(std::size_t h, std::size_t w) {
  if (width_ != w || height_ != h) {
    FIBITMAP* bitmap = FreeImage_Rescale(bitmap_, h, w, FILTER_BILINEAR);
    if (!bitmap) {
      throw std::runtime_error("error resizing image");
    }

    FreeImage_Unload(bitmap_);
    bitmap_ = bitmap;

    width_ = w;
    height_ = h;
  }

  // Do nothing if dimensions matches
}

void BitmapLoader::resizeV2(std::size_t h, std::size_t w) {
  assert(h == 224 && w == 224); // Only support 3 * 224 * 224
  if (height_ == h && width_ == w) {
    // Do nothing if dimensions matches
    return;
  }

  int resize_w, resize_h;
  if (height_ > width_) {
    resize_w = 256;
    resize_h = 256.0 * height_ / width_;
  } else {
    resize_h = 256;
    resize_w = 256.0 * width_ / height_;
  }

  FIBITMAP* bitmap =
    FreeImage_Rescale(bitmap_, resize_w, resize_h, FILTER_BILINEAR);
  if (!bitmap) {
    throw std::runtime_error("error resizing image");
  }

  int left = (resize_w - 224) / 2.0;
  int right = (resize_w + 224) / 2.0;
  int top = (resize_h - 224) / 2.0;
  int bottom = (resize_h + 224) / 2.0;

  FIBITMAP* bitmap_crop = FreeImage_Copy(bitmap, left, top, right, bottom);
  FreeImage_Unload(bitmap);
  if (!bitmap_crop) {
    throw std::runtime_error("error copy image");
  }

  std::size_t height = FreeImage_GetHeight(bitmap_crop);
  std::size_t width = FreeImage_GetWidth(bitmap_crop);
  assert(height == h && width == w);

  FreeImage_Unload(bitmap_);
  bitmap_ = bitmap_crop;

  width_ = w;
  height_ = h;
}

const uint8_t* BitmapLoader::scanLine(std::size_t h) {
  if (!bitmap_) {
    throw std::runtime_error("Image not initialized");
  }

  // Reverse scanline order
  return FreeImage_GetScanLine(bitmap_, static_cast<int>(height_ - h - 1));
}

template<class _V>
std::vector<_V> BitmapLoader::decode(bool is_int8) {
  std::vector<_V> decoded;
  decoded.resize(k_channel * height_ * width_);

  for (std::size_t i = 0; i < height_; ++i) {
    auto line = scanLine(i);
    for (std::size_t j = 0; j < width_; ++j) {
      auto idx = width_ * i + j;
      if (is_int8) {
        if (isGray()) {
          decoded[idx] = static_cast<_V>(*(line + j) - 123.68);
          decoded[idx + height_ * width_] =
            static_cast<_V>(*(line + j) - 116.78);
          decoded[idx + 2 * height_ * width_] =
            static_cast<_V>(*(line + j) - 103.94);
        } else {
          decoded[idx] = static_cast<_V>(*(line + j * 3 + 2) - 123.68);
          decoded[idx + height_ * width_] =
            static_cast<_V>(*(line + j * 3 + 1) - 116.78);
          decoded[idx + 2 * height_ * width_] =
            static_cast<_V>(*(line + j * 3) - 103.94);
        }
      } else {
        if (isGray()) {
          decoded[idx] = static_cast<_V>((*(line + j) / 255.0 - 0.485) / 0.229);
          decoded[idx + height_ * width_] =
            static_cast<_V>((*(line + j) / 255.0 - 0.456) / 0.224);
          decoded[idx + 2 * height_ * width_] =
            static_cast<_V>((*(line + j) / 255.0 - 0.406) / 0.225);
        } else {
          decoded[idx] =
            static_cast<_V>((*(line + j * 3 + 2) / 255.0 - 0.485) / 0.229);
          decoded[idx + height_ * width_] =
            static_cast<_V>((*(line + j * 3 + 1) / 255.0 - 0.456) / 0.224);
          decoded[idx + 2 * height_ * width_] =
            static_cast<_V>((*(line + j * 3) / 255.0 - 0.406) / 0.225);
        }
      }
    }
  }

  return decoded;
}

template<class _V>
std::vector<_V> BitmapLoader::decodeOrigin() {
  std::vector<_V> decoded;
  decoded.resize(k_channel * height_ * width_);

  for (std::size_t i = 0; i < height_; ++i) {
    auto line = scanLine(i);
    for (std::size_t j = 0; j < width_; ++j) {
      auto idx = width_ * i + j;
      if (isGray()) {
        decoded[idx] = static_cast<_V>((*(line + j)));
        decoded[idx + height_ * width_] = static_cast<_V>((*(line + j)));
        decoded[idx + 2 * height_ * width_] = static_cast<_V>((*(line + j)));
      } else {
        decoded[idx] = static_cast<_V>((*(line + j * 3 + 2)));
        decoded[idx + height_ * width_] =
          static_cast<_V>((*(line + j * 3 + 1)));
        decoded[idx + 2 * height_ * width_] =
          static_cast<_V>((*(line + j * 3)));
      }
    }
  }

  return decoded;
}

} // namespace suinfer
