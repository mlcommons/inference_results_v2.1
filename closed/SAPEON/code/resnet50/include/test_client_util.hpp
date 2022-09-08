/**
 * @file test_client_util.hpp
 * @author Heecheol Yang (heecheol.yang@sk.com)
 * @brief Utilities for example codes.
 * @version 0.1
 * @date 2021-09-27
 *
 * @copyright Copyright (c) 2021 SK TELECOM CO., LTD.
 *
 */
/**
 * @warning Do not use this header in actual application.
 */

#pragma once
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <condition_variable>
#include <exception>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace client_util {
template <typename T>
static std::shared_ptr<std::vector<T>> ReadFromFile(
    const std::string& fileName) {
  std::ifstream if_file(fileName, std::ios::binary);
  EXPECT_TRUE(if_file.is_open());
  if_file.seekg(0, std::ios::end);
  size_t file_size = if_file.tellg();
  if_file.seekg(0, std::ios::beg);
  auto result = std::make_shared<std::vector<T>>();
  result->resize(file_size / sizeof(T));
  if_file.read(reinterpret_cast<char*>(result->data()), file_size);
  return result;
}

template <typename T>
class Queue {
 public:
  void Push(const T& in) {
    std::unique_lock<std::mutex> l(lock_);
    queue_.push_back(in);
  }
  T Pop() {
    std::unique_lock<std::mutex> l(lock_);
    if (queue_.empty() == true) {
      // throw std::exception();
      return nullptr;
    }
    T ret = queue_.front();
    queue_.pop_front();
    return ret;
  }

  bool Empty() {
    std::unique_lock<std::mutex> l(lock_);
    return queue_.empty();
  }

  size_t Size() {
    std::unique_lock<std::mutex> l(lock_);
    return queue_.size();
  }

 private:
  std::list<T> queue_;
  std::mutex lock_;
};

template <typename T>
class PtrQueue {
 public:
  PtrQueue() {}
  PtrQueue(const PtrQueue&) = delete;
  PtrQueue& operator=(const PtrQueue&) = delete;

  void Push(std::unique_ptr<T>&& ptr) {
    std::lock_guard<std::mutex> lk(mut_);
    unique_ptr_q_.push(std::move(ptr));
    cv_.notify_one();
  }

  std::unique_ptr<T> Pop(const uint64_t timeout_ms = 100) {
    using namespace std::chrono_literals;  // NOLINT
    std::unique_lock<std::mutex> lk(mut_);
    if (unique_ptr_q_.empty() == true) {
      bool status = cv_.wait_for(lk, timeout_ms * 1ms,
                                 [this] { return !unique_ptr_q_.empty(); });
      if (status == false) {
        // timed out
        return nullptr;
      }
    }
    auto res = std::move(unique_ptr_q_.front());
    unique_ptr_q_.pop();
    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut_);
    return unique_ptr_q_.empty();
  }

  size_t Size() {
    std::lock_guard<std::mutex> lk(mut_);
    return unique_ptr_q_.size();
  }

 private:
  std::mutex mut_;
  std::queue<std::unique_ptr<T>> unique_ptr_q_;
  std::condition_variable cv_;
};

template <typename T>
class BufferedQueue {
 public:
  BufferedQueue(const size_t timeout_ms, const size_t max_items)
      : kTimeOutMs_(timeout_ms), kMaxItems_(max_items) {}
  void Push(const T& value) {
    std::unique_lock<std::mutex> lk(queue_m_);
    queue_.push(value);
    queue_cv_.notify_one();
  }
  std::vector<T> Pop() {
    using namespace std::chrono_literals;  // NOLINT
    std::unique_lock<std::mutex> lk(queue_m_);
    queue_cv_.wait_for(lk, kTimeOutMs_ * 1ms, [this] {
      return (queue_.size() >= kMaxItems_);
    });
    const size_t kPopSize = std::min(queue_.size(), kMaxItems_);
    std::vector<T> result(kPopSize);
    for (size_t repeat = 0; repeat < kPopSize; repeat++) {
      result[repeat] = queue_.front();
      queue_.pop();
    }
    return result;
  }

 private:
  const size_t kTimeOutMs_;
  const size_t kMaxItems_;
  std::queue<T> queue_;
  std::mutex queue_m_;
  std::condition_variable queue_cv_;
};
};  // namespace client_util
