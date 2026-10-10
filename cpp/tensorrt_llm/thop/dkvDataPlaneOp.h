/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/thop/ncclCommunicatorOp.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace th = torch;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! Carries out the data-plane program of one rank of a DKV layer-split group (dkv_streamer.py).
//!
//! The Python streamer compiles the plan of an iteration into the tables below; every hook of the
//! forward pass then hands one issue point to `run`, which enqueues the work of its steps on the data
//! stream: the copies between the pages of the cache manager, the staging slots and the message
//! arenas, and the NCCL group of the messages of each step. The tables are CPU int64 tensors:
//!
//!   components      [C, 3]          kind, attention type, page bytes
//!   poolAffine      [L, A, 3]       layer group (-1: this rank does not own the buffer), address step and
//!                                   address origin of the pages of the cache manager: block `b` of a request
//!                                   is at origin + base[b] * step, `base` being its base page indices
//!   stagingPointers [L, C]          the address the pages of the layer's slot are indexed from (0: none)
//!   slotOffsets     [L, K]          the pages to add to a page index of the layer's slot
//!   requests        [S, 1 + 2G + 3K] request id; per layer group the host address and the length of the
//!                                   int32 base page indices of the request (0, 0: none); per kind the first
//!                                   staged block, its page in the slot and the staged page count (-1: the
//!                                   request is not staged on this rank)
//!   templates       [T + 1]         where each template starts in `runs`
//!   runs            [R, 5]          component, request, first block, block count, byte offset in the message
//!   steps           [N, 6]          issue point, direction (0 fetch, 1 writeback), layer, record the fetch
//!                                   event of the layer, first op, op count
//!   ops             [M, 5]          action (0 send, 1 receive, 2 local copy), template, peer, offset of the
//!                                   message in its arena, bytes
//!
//! A template is the runs of the messages of one kind of transfer, which are the same in every layer of a
//! layer type; an op places it in a layer and a message. The base page indices are read on the host while
//! `run` builds the copies, so they must stay alive until the last `run` of the iteration.
class DkvDataPlaneOp : public th::jit::CustomClassHolder
{
public:
    DkvDataPlaneOp(c10::intrusive_ptr<NcclCommunicatorOp> const& comm, int64_t numLayers, int64_t dataStream);
    ~DkvDataPlaneOp() override;

    DkvDataPlaneOp(DkvDataPlaneOp const&) = delete;
    DkvDataPlaneOp& operator=(DkvDataPlaneOp const&) = delete;

    //! The tables that do not change between iterations.
    void setStatic(th::Tensor components, th::Tensor poolAffine, th::Tensor stagingPointers, th::Tensor slotOffsets);

    //! The program of the next iteration; `sendArena` and `recvArena` are the device addresses of the arenas.
    void setIteration(th::Tensor requests, th::Tensor templates, th::Tensor runs, th::Tensor steps, th::Tensor ops,
        int64_t sendArena, int64_t recvArena);

    //! Enqueue the steps of `issuePoint` (-1: before the first layer, L: after the last one). Unless the issue
    //! point is layer 0, the data stream first waits for the work enqueued on `computeStream` so far. For a
    //! layer whose fetch was recorded, `computeStream` then waits for it; `timeFetchWaits` brackets that wait
    //! with timing events.
    void run(int64_t issuePoint, int64_t computeStream, bool timeFetchWaits);

    //! The milliseconds the compute stream waited for fetches since the last call, once the timed waits are done.
    double fetchWaitMilliseconds();

private:
    using CopyTask = tensorrt_llm::batch_manager::kv_cache_manager_v2::Task<
        tensorrt_llm::batch_manager::kv_cache_manager_v2::MemAddress,
        tensorrt_llm::batch_manager::kv_cache_manager_v2::MemAddress>;

    //! Where the pages of a run are: page `i` at base + (indices ? indices[i] : i) * step.
    struct Side
    {
        int64_t base;
        int64_t step;
        int32_t const* indices;
    };

    struct Message
    {
        int64_t address;
        int64_t bytes;
        int peer;
    };

    void issueStep(size_t index);
    Side poolSide(int64_t layer, int64_t const* run) const;
    Side stagingSide(int64_t layer, int64_t const* run) const;
    Side messageSide(int64_t base, int64_t const* run) const;
    void addCopies(int64_t const* run, int64_t layer, Side const& destination, Side const& source);
    void launchCopies(char const* what, int64_t layer);
    void exchange(std::vector<Message> const& sends, std::vector<Message> const& receives) const;
    int64_t requestId(int64_t request) const;

    std::shared_ptr<tensorrt_llm::runtime::NcclCommunicator> mComm;
    int mRank;
    int mDevice;
    int64_t mNumLayers;
    cudaStream_t mDataStream;

    int64_t mNumComponents = 0;
    int64_t mNumAttentionTypes = 0;
    int64_t mNumKinds = 0;
    std::vector<int64_t> mComponents;
    std::vector<int64_t> mPoolAffine;
    std::vector<int64_t> mStagingPointers;
    std::vector<int64_t> mSlotOffsets;
    bool mStaticSet = false;

    int64_t mNumGroups = 0;
    int64_t mRequestWidth = 0;
    std::vector<int64_t> mRequests;
    std::vector<int64_t> mTemplates;
    std::vector<int64_t> mRuns;
    std::vector<int64_t> mSteps;
    std::vector<int64_t> mOps;
    //! The steps of issue point p are [mStepBegin[p + 1], mStepBegin[p + 2]).
    std::vector<size_t> mStepBegin;
    int64_t mSendArena = 0;
    int64_t mRecvArena = 0;

    //! One event per issue point that the data stream waits for, and one per layer for its fetch.
    std::vector<cudaEvent_t> mComputeEvents;
    std::vector<cudaEvent_t> mFetchEvents;
    std::vector<char> mFetchRecorded;
    std::vector<cudaEvent_t> mWaitStarts;
    std::vector<cudaEvent_t> mWaitEnds;
    std::vector<char> mWaitTimed;

    //! The copies of the part of a step being built, by page size.
    std::vector<std::pair<int64_t, std::vector<CopyTask>>> mCopies;
    std::vector<Message> mSends;
    std::vector<Message> mReceives;
};

//! Registers the class as trtllm::DkvDataPlane. Torch looks up the NcclCommunicatorOp its constructor takes when the
//! class registers, so ncclCommunicatorOp.cpp calls this after it registers that class.
th::jit::class_<DkvDataPlaneOp> registerDkvDataPlane();

} // namespace torch_ext

TRTLLM_NAMESPACE_END
