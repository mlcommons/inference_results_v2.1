/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

// Copyright (C) 2022 Alibaba Group Holding Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "bert_server.h"
#include "bert_core_vs.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>

template <typename T>
void BERTServer::ProcessTasks(std::shared_ptr<T> bertCore, int deviceId, int qThreadIdx)
{
    uint64_t totalCountInThread = 0;
    bertCore->SetActiveDevice();
    // hold soft drop tasks if any
    BERTTask_t holdedTasks;

    // Process samples in batches
    auto tasks = GetTasks(mMaxBatchSize, qThreadIdx);

    while (!tasks.empty())
    {
        totalCountInThread += tasks.size();
        if (mSoftDrop < 1.0)
        {
            std::unique_lock<std::mutex> lck(mSoftDropMtx);
            mTotalTasksCount += tasks.size();

            mTotalLengthSet.InsertTasks(tasks, mQsl);

            // Drop requests until the total length is not greater than the threshold
            // Use target latency percentile as a hard limit on how many requests we can drop
            while (BERTCoreVS::CountTotalLength(tasks, mQsl) > mTotalLengthSet.GetThresholdLength()
                && mSoftDropCount
                    <= std::floor(static_cast<double>(mTotalTasksCount) * (1.0 - mTargetLatencyPercentile)) - 1)
            {
                holdedTasks.push_back(tasks.front());
                tasks.erase(tasks.begin());
                ++mSoftDropCount;
            }
        }
        bertCore->infer(tasks, mQsl);
        tasks = GetTasks(mMaxBatchSize, qThreadIdx);
    }

    if (mSoftDrop < 1.0)
    {
        // Process soft drop tasks if any
        LOG(INFO) << "Total number of soft drop tasks: " << holdedTasks.size() << " out of " << totalCountInThread
                  << " total tasks";
        while (holdedTasks.size() != 0)
        {
            std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>> tasks;
            tasks.reserve(mMaxBatchSize);
            // Consume up to mMaxBatchSize tasks
            for (int i = 0; (i < mMaxBatchSize) && !holdedTasks.empty(); ++i)
            {
                tasks.push_back(holdedTasks.back());
                holdedTasks.pop_back();
            }
            bertCore->infer(tasks, mQsl);
        }
        // This is necessary to avoid a race condition if the bertCore is destructed before we
        // process all responses
        bertCore->WaitUntilQueueEmpty();
    }

    using CLK = std::chrono::high_resolution_clock;
    VLOG(1) << "End of ProcessTasks: "
            << std::chrono::duration_cast<std::chrono::microseconds>(CLK::now().time_since_epoch()).count();
}

void BERTServer::StartIssueThread(int threadIdx)
{
    {
        CHECK_EQ(!mMtxs->empty(), true);
        std::lock_guard<std::mutex> lock((*mMtxs)[0]);
        mThreadMap[std::this_thread::get_id()] = threadIdx;
    }
    mlperf::RegisterIssueQueryThread();
}

BERTServer::BERTServer(const std::string name, const std::string enginePath, std::shared_ptr<qsl::SampleLibrary> qsl,
    const std::vector<int>& gpus, int maxBatchSize, int numCopyStreams, int numBERTCores, bool useGraphs,
    int graphMaxSeqLen, const std::string& graphSpecs, double softDrop, double targetLatencyPercentile,
    uint64_t serverNumIssueQueryThreads)
    : mName{name}
    , mQsl{qsl}
    , mStopGetTasks{false}
    , mStopProcessResponse{false}
    , mMaxBatchSize{maxBatchSize}
    , mGraphMaxSeqLen{graphMaxSeqLen}
    , mSoftDrop{softDrop}
    , mTargetLatencyPercentile{targetLatencyPercentile}
    , mTotalLengthSet{mSoftDrop}
    , mTotalTasksCount{0}
    , mSoftDropCount{0}
{
    LOG(INFO) << odla_GetVersionString();
    LOG(INFO) << "Engines Creation Completed";

    if (useGraphs)
    {
        LOG(INFO) << "Use CUDA graphs";
    }

    LOG(INFO) << "numBERTCores:" << numBERTCores;
    using BERTCoreVSPtrVec = std::vector<std::shared_ptr<BERTCoreVS>>;
    std::vector<BERTCoreVSPtrVec> tmpBERTCores(gpus.size());
    for (int idx = 0; idx < gpus.size(); ++idx)
    {
        auto deviceId = gpus[idx];
        tmpBERTCores[idx].push_back(std::make_shared<BERTCoreVS>(
            enginePath, numCopyStreams, numBERTCores, 0, useGraphs, deviceId, mMaxBatchSize));

        for (int profileIdx = 1; profileIdx < numBERTCores; ++profileIdx)
        {
            tmpBERTCores[idx].push_back(std::make_shared<BERTCoreVS>(*tmpBERTCores[idx].front(), profileIdx));
        }
    }

    if (mSoftDrop < 1.0)
    {
        LOG(INFO) << "Apply soft drop policy with threshold = " << mSoftDrop;
    }

    LOG(INFO) << "serverNumIssueQueryThreads:" << serverNumIssueQueryThreads;
    if (serverNumIssueQueryThreads > 0)
    {
        CHECK_EQ((gpus.size() * numBERTCores) % serverNumIssueQueryThreads == 0, true);
        LOG(INFO) << "Use number of server IssueQuery threads = " << serverNumIssueQueryThreads;
        mTasksVec.resize(serverNumIssueQueryThreads);
        mMtxs = std::make_unique<std::vector<std::mutex>>(serverNumIssueQueryThreads);
        mCondVars = std::make_unique<std::vector<std::condition_variable>>(serverNumIssueQueryThreads);
        for (int i = 0; i < serverNumIssueQueryThreads; ++i)
        {
            mIssueQueryThreads.emplace_back(&BERTServer::StartIssueThread, this, i);
        }
    }
    else
    {
        mTasksVec.resize(1);
        mMtxs = std::make_unique<std::vector<std::mutex>>(1);
        mCondVars = std::make_unique<std::vector<std::condition_variable>>(1);
    }

    // Warm up BERTCoreVS and launch threads for processing tasks
    int BERTCoresPerQThread
        = (serverNumIssueQueryThreads == 0) ? INT_MAX : (gpus.size() * numBERTCores) / serverNumIssueQueryThreads;
    int counter = 0;
    int qThreadIdx = 0;

    for (int idx = 0; idx < gpus.size(); ++idx)
    {
        for (auto& bertCore : tmpBERTCores[idx])
            bertCore->WarmUp();
    }

    mWorkerThreads.reserve(gpus.size() * numBERTCores);
    for (int idx = 0; idx < gpus.size(); ++idx)
    {
        auto deviceId = gpus[idx];
        for (auto& bertCore : tmpBERTCores[idx])
        {
            CHECK_EQ(mMaxBatchSize <= bertCore->GetMaxBatchSize(), true);
            mWorkerThreads.emplace_back(&BERTServer::ProcessTasks<BERTCoreVS>, this, bertCore, deviceId, qThreadIdx);

            ++counter;
            if (counter == BERTCoresPerQThread)
            {
                ++qThreadIdx;
                counter = 0;
            }
        }
    }
}

BERTServer::~BERTServer()
{
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i)
        {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        mStopProcessResponse = true;
        for (int i = 0; i < mCondVars->size(); ++i)
        {
            (*mCondVars)[i].notify_all();
        }
    }
    for (auto& workerThread : mWorkerThreads)
    {
        workerThread.join();
    }
    for (auto& issueQueryThread : mIssueQueryThreads)
    {
        issueQueryThread.join();
    }
    LOG(INFO) << "BERT Server stopped";
}

const std::string& BERTServer::Name() const
{
    return mName;
}

void BERTServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    auto queryArrivedTime = std::chrono::high_resolution_clock::now();

    // Sort samples in the descending order of sentence length
    std::vector<std::pair<int, int>> sequenceSamplePosAndLength(samples.size());
    for (int samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        sequenceSamplePosAndLength[samplePos]
            = std::make_pair(samplePos, static_cast<int>(GetSampleLength(samples[samplePos].index)));
    }

    std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) -> bool { return a.second > b.second; });

    int qThreadIdx = mThreadMap[std::this_thread::get_id()];
    for (int beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size(); beginSamplePos += mMaxBatchSize)
    {
        int actualBatchSize
            = std::min(mMaxBatchSize, static_cast<int>(sequenceSamplePosAndLength.size()) - beginSamplePos);
        static int totalBatchSize = 0;
        totalBatchSize += actualBatchSize;
        {
            std::unique_lock<std::mutex> lck((*mMtxs)[qThreadIdx]);
            for (int i = 0; i < actualBatchSize; ++i)
            {
                int samplePosInOriginalRequest = sequenceSamplePosAndLength[beginSamplePos + i].first;
                mTasksVec[qThreadIdx].push_back({samples[samplePosInOriginalRequest], queryArrivedTime});
            }

            // Let some worker thread to consume tasks
            (*mCondVars)[qThreadIdx].notify_one();
        }
    }
}

void BERTServer::FlushQueries()
{
    if (mSoftDrop < 1.0)
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (int i = 0; i < mMtxs->size(); ++i)
        {
            lcks.emplace_back((*mMtxs)[i]);
        }
        mStopGetTasks = true;
        for (int i = 0; i < mCondVars->size(); ++i)
        {
            (*mCondVars)[i].notify_all();
        }
    }
}

std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>> BERTServer::GetTasks(
    int maxSampleCount, int qThreadIdx)
{
    std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>> res;
    res.reserve(maxSampleCount);
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck((*mMtxs)[qThreadIdx]);
    (*mCondVars)[qThreadIdx].wait(lck, [&] { return (!mTasksVec[qThreadIdx].empty()) || mStopGetTasks; });

    // Consume up to maxSampleCount tasks
    for (int i = 0; (i < maxSampleCount) && !mTasksVec[qThreadIdx].empty(); ++i)
    {
        res.push_back(mTasksVec[qThreadIdx].front());
        mTasksVec[qThreadIdx].pop_front();
    }

    // Let some other thread consume remaining tasks
    if (!mTasksVec[qThreadIdx].empty())
    {
        (*mCondVars)[qThreadIdx].notify_one();
    }

    return res;
}

size_t BERTServer::GetSampleLength(mlperf::QuerySampleIndex idx)
{
    // Get sample length by checking where the input_mask change from 1 to 0
    size_t start{0};
    size_t end{BERT_MAX_SEQ_LENGTH};
    size_t cursor{(start + end) / 2};
    BERTInput input_mask = *static_cast<BERTInput*>(mQsl->GetSampleAddress(idx, 2));
    while (cursor != start)
    {
        if (input_mask[cursor])
        {
            start = cursor;
        }
        else
        {
            end = cursor;
        }
        cursor = (start + end) / 2;
    }
    return end;
}
