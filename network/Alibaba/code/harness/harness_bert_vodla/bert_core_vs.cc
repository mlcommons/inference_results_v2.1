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

#include <ODLA/odla.h>

#include "bert_core_vs.h"

#include "bert_server.h"
#include "glog/logging.h"
#include "loadgen.h"

#include <fstream>
#include <set>
#include <unordered_set>

#undef CUDA_GRAPH_STATS

constexpr int BIDX = 0;
constexpr int BERT_CUDA_GRAPH_SIZE = 43; // in MiB ranges from ~[38,45]

void BERTCoreVS::SetInputShapes(odla_context context, int sumS, int B, int maxS)
{
    if (sumS != mSumS)
    {
        CHECK_EQ(odla_SetRuntimeValueType(context, mInputIds, {ODLA_INT32, {1, {sumS}}}), ODLA_SUCCESS);
        CHECK_EQ(odla_SetRuntimeValueType(context, mSegmentIds, {ODLA_INT32, {1, {sumS}}}), ODLA_SUCCESS);
        mSumS = sumS;
    }
    if (B != mB)
    {
        CHECK_EQ(odla_SetRuntimeValueType(context, mCuSeqLens, {ODLA_INT32, {1, {B + 1}}}), ODLA_SUCCESS);
        mB = B;
    }
    if (maxS != mMaxS)
    {
        CHECK_EQ(odla_SetRuntimeValueType(context, mMaxSeqLens, {ODLA_INT32, {1, {maxS}}}), ODLA_SUCCESS);
        mMaxS = maxS;
    }
}

BERTCoreVS::BERTCoreVS(const std::string& enginePath, int numCopyStreams, int numProfiles, int profileIdx,
    bool useGraphs, int deviceId, int maxBS)
    : mCopyStreams(numCopyStreams)
    , mCounter(0)
    , mUseGraphs(useGraphs)
    , mDeviceId(deviceId)
    , mCtx(nullptr)
    , mIdx(profileIdx)
    , mMaxBatchSize(maxBS)
    , mResLoc{ODLA_LOCATION_PATH, enginePath.c_str(), enginePath.size()}
{
    odla_device d;
    CHECK(odla_AllocateDevice(nullptr, ODLA_DEVICE_DEFAULT, deviceId, &d) == ODLA_SUCCESS);
    mDevice.reset(d, [](odla_device dev) { odla_DestroyDevice(dev); });
    LOG(INFO) << "Create BERTCoreVS (device: " << mDeviceId << " tid:" << std::this_thread::get_id() << ")";

    CHECK(odla_LoadExecutable(mResLoc, mDevice.get(), &mExec) == ODLA_SUCCESS);
    CHECK(odla_CreateContext(&mCtx) == ODLA_SUCCESS);
    for (int it = 0; it < NUM_RESPONSE_THREADS; it++)
    {
        mResponseThreads.emplace_back(BERTCoreVS::ProcessResponse, this);
    }

    int maxNumBindings = 0;

    unsigned num_inputs = 0;
    unsigned num_outputs = 0;

    odla_GetNumOfArgsFromExecutable(mExec, &num_inputs);
    // odla_GetNumOfOutputsFromExecutable(odla_exec, &num_outputs);
    num_outputs = 1;
    maxNumBindings = num_inputs + num_outputs;
    odla_GetArgFromExecutableByIdx(mExec, 0, &mInputIds);
    odla_GetArgFromExecutableByIdx(mExec, 1, &mSegmentIds);
    odla_GetArgFromExecutableByIdx(mExec, 2, &mCuSeqLens);
    odla_GetArgFromExecutableByIdx(mExec, 3, &mMaxSeqLens);
    odla_GetOutputFromExecutableByIdx(mExec, 0, &mOutput);

    mMaxDimsToContext.insert({{mMaxBatchSize, 384}, nullptr});

    LOG(INFO) << "maxNumBindings:" << maxNumBindings;
    CHECK_EQ(mMaxBatchSize > 0, true);
    LOG(INFO) << "Context creation complete. Max supported batchSize: " << mMaxBatchSize;

    // Allocate buffers
    const size_t bufferSize = BERT_MAX_SEQ_LENGTH * mMaxBatchSize;

    CHECK_EQ(maxNumBindings % 5, 0);
    for (int it = 0; it < mCopyStreams.size(); it++)
    {
        mInputIdBufs.emplace_back(bufferSize);
        mSegmentIdBufs.emplace_back(bufferSize);
        // TODO need to rename this
        mInputMaskBufs.emplace_back(mMaxBatchSize + 1);
        mDummy.emplace_back(BERT_MAX_SEQ_LENGTH);
        mOutputBufs.emplace_back(BERT_MAX_SEQ_LENGTH * mMaxBatchSize * 2);
    }

    LOG(INFO) << "Setup complete";
}

BERTCoreVS::BERTCoreVS(const BERTCoreVS& other, int idx)
    : mIdx(idx)
    , mDevice(other.mDevice)
    //, mExec(other.mExec)
    , mResLoc(other.mResLoc)
    , mCopyStreams(other.mCopyStreams.size())
    , mMaxBatchSize(other.mMaxBatchSize)
    , mCounter(other.mCounter)
    , mUseGraphs(other.mUseGraphs)
    , mDeviceId(other.mDeviceId)
    //, mInputIds(other.mInputIds)
    //, mSegmentIds(other.mSegmentIds)
    //, mCuSeqLens(other.mCuSeqLens)
    //, mMaxSeqLens(other.mMaxSeqLens)
    , mOutput(other.mOutput)
{

    LOG(INFO) << "Create BERTCoreVS (device: " << mDeviceId << " tid:" << std::this_thread::get_id() << ")";

    CHECK(odla_LoadExecutable(mResLoc, mDevice.get(), &mExec) == ODLA_SUCCESS);
    CHECK(odla_CreateContext(&mCtx) == ODLA_SUCCESS);
    CHECK(odla_GetArgFromExecutableByIdx(mExec, 0, &mInputIds) == ODLA_SUCCESS);
    CHECK(odla_GetArgFromExecutableByIdx(mExec, 1, &mSegmentIds) == ODLA_SUCCESS);
    CHECK(odla_GetArgFromExecutableByIdx(mExec, 2, &mCuSeqLens) == ODLA_SUCCESS);
    CHECK(odla_GetArgFromExecutableByIdx(mExec, 3, &mMaxSeqLens) == ODLA_SUCCESS);
    CHECK(odla_GetOutputFromExecutableByIdx(mExec, 0, &mOutput) == ODLA_SUCCESS);

    for (int it = 0; it < NUM_RESPONSE_THREADS; it++)
    {
        mResponseThreads.emplace_back(BERTCoreVS::ProcessResponse, this);
    }

    mMaxDimsToContext.insert({{mMaxBatchSize, 384}, nullptr});

    LOG(INFO) << "Context creation complete. Max supported batchSize: " << mMaxBatchSize;

    // Allocate buffers
    const size_t bufferSize = BERT_MAX_SEQ_LENGTH * mMaxBatchSize;

    for (int it = 0; it < mCopyStreams.size(); it++)
    {
        mInputIdBufs.emplace_back(bufferSize);
        mSegmentIdBufs.emplace_back(bufferSize);
        // TODO need to rename this
        mInputMaskBufs.emplace_back(mMaxBatchSize + 1);
        mDummy.emplace_back(BERT_MAX_SEQ_LENGTH);
        mOutputBufs.emplace_back(BERT_MAX_SEQ_LENGTH * mMaxBatchSize * 2);
    }

    LOG(INFO) << "Setup complete for device " << mDeviceId << " thread:" << mIdx;
}

odla_context BERTCoreVS::GetContext(int batchSize, int seqLen)
{
    return mCtx;
}

int BERTCoreVS::GetClosestSeqLen(int seqLen)
{
    // find the first context whose max batch size is larger or equal to the given one
    auto current = mMaxDimsToContext.begin();
    while (current != mMaxDimsToContext.end() && seqLen > current->first.second)
    {
        current++;
    }
    CHECK_EQ(current == mMaxDimsToContext.end(), false);
    return current->first.second;
}

void BERTCoreVS::SetActiveDevice()
{
    odla_SetCurrentDevice(mDevice.get());
}

void BERTCoreVS::ProcessResponse(BERTCoreVS* BERTCoreVS)
{
    size_t totSamples = 0;
    while (true)
    {
        std::unique_lock<std::mutex> lck(BERTCoreVS->mMtx);
        BERTCoreVS->mCondVar.wait(lck, [&]() { return !BERTCoreVS->mResultQ.empty() || BERTCoreVS->mStopWork; });
        if (BERTCoreVS->mStopWork)
            break;
        auto& resp = BERTCoreVS->mResultQ.front();

        for (auto& qsr : resp.QSRs)
        {
            mlperf::QuerySamplesComplete(&qsr, 1);
        }
        totSamples += resp.QSRs.size();

        BERTCoreVS->mCopyStreamIdxQueue.push_back(resp.copyStreamIdx);
        BERTCoreVS->mResultQ.pop_front();
        BERTCoreVS->mCondVar.notify_one();
    }

    VLOG(1) << "QuerySamplesCompelete " << totSamples << " samples.";
    using CLK = std::chrono::high_resolution_clock;
    VLOG(1) << "End of ProcessResponse: "
            << std::chrono::duration_cast<std::chrono::microseconds>(CLK::now().time_since_epoch()).count();
}

int BERTCoreVS::CountTotalLength(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl)
{
    const int actualBatchSize = tasks.size();
    int totalLength = 0;
    for (int i = 0; i < actualBatchSize; ++i)
    {
        BERTInput* mask = static_cast<BERTInput*>(qsl->GetSampleAddress(tasks[i].first.index, 2));
        int Si = std::accumulate(mask->begin(), mask->end(), 0);
        totalLength += Si;
    }
    return totalLength;
}

void BERTCoreVS::infer(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl)
{
    const int actualBatchSize = tasks.size();
    static int totalBatchSize = 0;
    totalBatchSize += actualBatchSize;
    // iterate through the batch and find largest seqLen
    int maxSeqLen = 0;
    // accumulate the sequence lengths
    std::vector<int> cuSeqlens(actualBatchSize + 1, 0);

    auto& inputIds = mInputIdBufs[0];
    auto& segmentIds = mSegmentIdBufs[0];
    auto& inputMask = mInputMaskBufs[0];
    auto& cuSeqLens = mDummy[0];
    auto& outputBuf = mOutputBufs[0];

    for (int i = 0; i < actualBatchSize; ++i)
    {
        BERTInput* mask = static_cast<BERTInput*>(qsl->GetSampleAddress(tasks[i].first.index, 2));
        int Si = std::accumulate(mask->begin(), mask->end(), 0);
        cuSeqlens[i + 1] = cuSeqlens[i] + Si;
        maxSeqLen = std::max(maxSeqLen, Si);
    }
    // find the closest one that is supported
    int seqLen = this->GetClosestSeqLen(maxSeqLen);
    VLOG(1) << "Max SeqLen found in Batch: " << maxSeqLen << " chosen: " << seqLen;

    // stage batch using correct batchSize and seqLen
    for (int i = 0; i < actualBatchSize; ++i)
    {
        const int offset = cuSeqlens[i];                   // offset
        const int numElements = cuSeqlens[i + 1] - offset; // numElements
        inputIds.H2H(qsl->GetSampleAddress(tasks[i].first.index, 0), offset, numElements);
        segmentIds.H2H(qsl->GetSampleAddress(tasks[i].first.index, 1), offset, numElements);
    }

    inputMask.H2H(cuSeqlens.data(), 0, cuSeqlens.size());

    // size of packed sequences
    const size_t sumS = cuSeqlens.back();

    // Pad dummy values if using CUDA Graphs and there is a valid Graph Spec.
    int dummyBatchSize = 0;
    std::vector<int> dummySeqlens;
    bool launchGraph = false;
    // a BERTCoreVS object per thread, so no contention for its resources
    // Set batch size
    CHECK_EQ(actualBatchSize <= mMaxBatchSize, true);

    VLOG(2) << "MaxSeqlen: " << seqLen << " Input length: " << cuSeqlens.back() << " Batch size: " << actualBatchSize;

    CHECK_EQ(odla_BindToArgument(mInputIds, inputIds.HostData(), mCtx), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mSegmentIds, segmentIds.HostData(), mCtx), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mCuSeqLens, inputMask.HostData(), mCtx), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mMaxSeqLens, cuSeqLens.HostData(), mCtx), ODLA_SUCCESS);

    SetInputShapes(mCtx, sumS, actualBatchSize, maxSeqLen);

    // Run inference

    CHECK_EQ(odla_LaunchExecutable(mExec, mCtx), ODLA_SUCCESS);

    CHECK_EQ(odla_GetValueData(mOutput, outputBuf.HostData(), mCtx), ODLA_SUCCESS);

    // prepare the response
    BERTResponse resp;
    char* ptr = reinterpret_cast<char*>(outputBuf.HostData());

    resp.QSRs.reserve(actualBatchSize);
    for (size_t i = 0; i < actualBatchSize; i++)
    {
        const int s_b = cuSeqlens[i + 1] - cuSeqlens[i];
        const size_t logitSizeInBytes = 2 * s_b * sizeof(BERTOutputType);
        // this is to handle warmup - is it robust?
        if (tasks[i].first.id == 0)
            continue;
        mlperf::QuerySampleResponse response{tasks[i].first.id, reinterpret_cast<uintptr_t>(ptr), logitSizeInBytes};
        resp.QSRs.emplace_back(response);
        resp.copyStreamIdx = mCounter;

        ptr += logitSizeInBytes;
    }

    EnqueueResponse(resp);
}

void BERTCoreVS::WarmUp()
{
    CHECK_EQ(mCounter, 0);
    odla_SetCurrentDevice(mDevice.get());
    auto context = mCtx;
    SetInputShapes(context, mMaxBatchSize * BERT_MAX_SEQ_LENGTH, mMaxBatchSize, BERT_MAX_SEQ_LENGTH);
    CHECK_EQ(odla_BindToArgument(mInputIds, mInputIdBufs.front().HostData(), context), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mSegmentIds, mSegmentIdBufs.front().HostData(), context), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mCuSeqLens, mInputMaskBufs.front().HostData(), context), ODLA_SUCCESS);
    CHECK_EQ(odla_BindToArgument(mMaxSeqLens, mDummy.front().HostData(), context), ODLA_SUCCESS);
    auto s = odla_LaunchExecutable(mExec, context);
    CHECK_EQ(s, ODLA_SUCCESS);
    if (s != ODLA_SUCCESS)
    {
        assert(0);
    }
    CHECK_EQ(odla_GetValueData(mOutput, mOutputBufs.front().HostData(), context), ODLA_SUCCESS);
    LOG(INFO) << "Done warmup (device: " << mDeviceId << " tid:" << std::this_thread::get_id() << ")";
}

BERTCoreVS::~BERTCoreVS()
{
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for (auto& rt : mResponseThreads)
    {
        rt.join();
    }

    static std::mutex m;
    {
        std::unique_lock<std::mutex> lck(m);

        odla_SetCurrentDevice(mDevice.get());
        odla_DestroyContext(mCtx);
        odla_DestroyExecutable(mExec);
        LOG(INFO) << "Destroy device " << mDevice;
        mDevice.reset();
    }
}
