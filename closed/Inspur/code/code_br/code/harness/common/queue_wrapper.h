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

#include "syncqueue.h"

#define QUEUE_TYPE SyncQueue
#define PUSH_QUEUE(q, element) (q).push_back(element)
#define POP_QUEUE(q) (q).front_then_pop()
#define ACQUIRE_QUEUE_TIMEOUT(q, element, timeout) (q).acquire(element, timeout, 1, true)