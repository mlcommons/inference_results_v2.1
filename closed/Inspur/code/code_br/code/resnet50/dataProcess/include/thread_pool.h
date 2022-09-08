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

#include <mutex>
#include <queue>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

template<typename T>
class SafeQueue {
public:
  SafeQueue() {}
  SafeQueue(SafeQueue&& other) {}
  ~SafeQueue() {}

  bool empty() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.empty();
  }

  int size() {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_queue.size();
  }

  void enqueue(T& t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_queue.emplace(t);
  }

  bool dequeue(T& t) {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_queue.empty()) {
      return false;
    }
    t = std::move(m_queue.front());
    m_queue.pop();
    return true;
  }

private:
  std::queue<T> m_queue;
  std::mutex m_mutex;
};

class ThreadPool {
public:
  ThreadPool(const int n_threads = 4)
    : m_threads(std::vector<std::thread>(n_threads)), m_shutdown(false) {}

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Inits thread pool
  void init() {
    for (size_t i = 0; i < m_threads.size(); ++i) {
      m_threads.at(i) = std::thread(ThreadWorker(this, i));
    }
  }

  void shutdown() {
    m_shutdown = true;
    m_conditional_lock.notify_all();

    for (size_t i = 0; i < m_threads.size(); ++i) {
      if (m_threads.at(i).joinable()) {
        m_threads.at(i).join();
      }
    }
  }

  template<typename F, typename... Args>
  auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    // Create a function with bounded parameter ready to execute
    std::function<decltype(f(args...))()> func =
      std::bind(std::forward<F>(f), std::forward<Args>(args)...);

    auto task_ptr =
      std::make_shared<std::packaged_task<decltype(f(args...))()>>(func);

    // Warp packaged task into void function
    std::function<void()> warpper_func = [task_ptr]() { (*task_ptr)(); };

    m_queue.enqueue(warpper_func);
    m_conditional_lock.notify_one();

    return task_ptr->get_future();
  }

private:
  class ThreadWorker {
  public:
    ThreadWorker(ThreadPool* pool, const int id) : m_id(id), m_pool(pool) {}

    void operator()() {
      std::function<void()> func;
      bool dequeued;

      while (!m_pool->m_shutdown) {
        {
          std::unique_lock<std::mutex> lock(m_pool->m_conditional_mutex);
          if (m_pool->m_queue.empty()) {
            m_pool->m_conditional_lock.wait(lock);
          }

          dequeued = m_pool->m_queue.dequeue(func);
        }

        if (dequeued)
          func();
      }
    }

  private:
    int m_id;
    ThreadPool* m_pool;
  };

  SafeQueue<std::function<void()>> m_queue;
  std::vector<std::thread> m_threads;
  bool m_shutdown;
  std::mutex m_conditional_mutex;
  std::condition_variable m_conditional_lock;
};
