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

#include "tensorrt_llm/thop/dkvDataPlaneOp.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <algorithm>
#include <iterator>

namespace tr = tensorrt_llm::runtime;
namespace kvc = tensorrt_llm::batch_manager::kv_cache_manager_v2;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kComponentFields = 3;
constexpr int64_t kAffineFields = 3;
constexpr int64_t kRunFields = 5;
constexpr int64_t kStepFields = 6;
constexpr int64_t kOpFields = 5;

constexpr int64_t kSend = 0;
constexpr int64_t kReceive = 1;
constexpr int64_t kLocal = 2;

//! The values of a CPU int64 tensor of `dims` dimensions; `shape` gives the sizes that must match (-1: any).
std::vector<int64_t> int64Table(th::Tensor const& tensor, std::vector<int64_t> const& shape, char const* name)
{
    TLLM_CHECK_WITH_INFO(
        tensor.device().is_cpu() && tensor.scalar_type() == th::kInt64, "%s must be a CPU int64 tensor", name);
    TLLM_CHECK_WITH_INFO(tensor.dim() == static_cast<int64_t>(shape.size()), "%s must have %zu dimensions, has %ld",
        name, shape.size(), static_cast<long>(tensor.dim()));
    for (size_t dim = 0; dim < shape.size(); ++dim)
    {
        TLLM_CHECK_WITH_INFO(shape[dim] < 0 || tensor.size(dim) == shape[dim],
            "dimension %zu of %s must be %ld, is %ld", dim, name, static_cast<long>(shape[dim]),
            static_cast<long>(tensor.size(dim)));
    }
    auto const contiguous = tensor.contiguous();
    auto const* data = contiguous.data_ptr<int64_t>();
    return std::vector<int64_t>(data, data + contiguous.numel());
}

void destroyEvents(std::vector<cudaEvent_t>& events)
{
    for (auto event : events)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
    events.clear();
}

std::vector<cudaEvent_t> createEvents(size_t count, unsigned int flags)
{
    std::vector<cudaEvent_t> events(count, nullptr);
    for (auto& event : events)
    {
        TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
    }
    return events;
}

} // namespace

DkvDataPlaneOp::DkvDataPlaneOp(
    c10::intrusive_ptr<NcclCommunicatorOp> const& comm, int64_t numLayers, int64_t dataStream)
    : mComm(comm->communicator())
    , mRank(comm->rank())
    , mDevice(0)
    , mNumLayers(numLayers)
    , mDataStream(reinterpret_cast<cudaStream_t>(dataStream))
{
    TLLM_CHECK_WITH_INFO(numLayers > 0, "the model needs at least one layer, got %ld", static_cast<long>(numLayers));
    TLLM_CUDA_CHECK(cudaGetDevice(&mDevice));
    mComputeEvents = createEvents(numLayers + 2, cudaEventDisableTiming);
    mFetchEvents = createEvents(numLayers, cudaEventDisableTiming);
    mFetchRecorded.assign(numLayers, 0);
    mWaitTimed.assign(numLayers, 0);
    mStepBegin.assign(numLayers + 3, 0);
}

DkvDataPlaneOp::~DkvDataPlaneOp()
{
    destroyEvents(mComputeEvents);
    destroyEvents(mFetchEvents);
    destroyEvents(mWaitStarts);
    destroyEvents(mWaitEnds);
}

void DkvDataPlaneOp::setStatic(
    th::Tensor components, th::Tensor poolAffine, th::Tensor stagingPointers, th::Tensor slotOffsets)
{
    TLLM_CHECK_WITH_INFO(components.dim() == 2, "components must be [components, 3]");
    TLLM_CHECK_WITH_INFO(poolAffine.dim() == 3, "poolAffine must be [layers, attention types, 3]");
    TLLM_CHECK_WITH_INFO(slotOffsets.dim() == 2, "slotOffsets must be [layers, kinds]");
    mNumComponents = components.size(0);
    mNumAttentionTypes = poolAffine.size(1);
    mNumKinds = slotOffsets.size(1);
    mComponents = int64Table(components, {mNumComponents, kComponentFields}, "components");
    mPoolAffine = int64Table(poolAffine, {mNumLayers, mNumAttentionTypes, kAffineFields}, "poolAffine");
    mStagingPointers = int64Table(stagingPointers, {mNumLayers, mNumComponents}, "stagingPointers");
    mSlotOffsets = int64Table(slotOffsets, {mNumLayers, mNumKinds}, "slotOffsets");
    for (int64_t component = 0; component < mNumComponents; ++component)
    {
        auto const* fields = &mComponents[component * kComponentFields];
        TLLM_CHECK_WITH_INFO(fields[0] >= 0 && fields[0] < mNumKinds && fields[1] >= 0 && fields[1] < mNumAttentionTypes
                && fields[2] > 0 && fields[2] % 16 == 0,
            "component %ld has kind %ld, attention type %ld and %ld page bytes", static_cast<long>(component),
            static_cast<long>(fields[0]), static_cast<long>(fields[1]), static_cast<long>(fields[2]));
    }
    mStaticSet = true;
}

void DkvDataPlaneOp::setIteration(th::Tensor requests, th::Tensor templates, th::Tensor runs, th::Tensor steps,
    th::Tensor ops, int64_t sendArena, int64_t recvArena)
{
    TLLM_CHECK_WITH_INFO(mStaticSet, "set_static must come before the first iteration");
    TLLM_CHECK_WITH_INFO(requests.dim() == 2, "requests must be [requests, 1 + 2 * groups + 3 * kinds]");
    mRequestWidth = requests.size(1);
    int64_t const groupFields = mRequestWidth - 1 - 3 * mNumKinds;
    TLLM_CHECK_WITH_INFO(groupFields >= 0 && groupFields % 2 == 0,
        "a request row has %ld fields, which is not 1 + 2 * groups + 3 * %ld kinds", static_cast<long>(mRequestWidth),
        static_cast<long>(mNumKinds));
    mNumGroups = groupFields / 2;
    mRequests = int64Table(requests, {-1, mRequestWidth}, "requests");
    mTemplates = int64Table(templates, {-1}, "templates");
    mRuns = int64Table(runs, {-1, kRunFields}, "runs");
    mSteps = int64Table(steps, {-1, kStepFields}, "steps");
    mOps = int64Table(ops, {-1, kOpFields}, "ops");
    mSendArena = sendArena;
    mRecvArena = recvArena;

    auto const numRequests = static_cast<int64_t>(mRequests.size()) / std::max<int64_t>(mRequestWidth, 1);
    auto const numRuns = static_cast<int64_t>(mRuns.size() / kRunFields);
    auto const numTemplates = static_cast<int64_t>(mTemplates.size()) - 1;
    auto const numOps = static_cast<int64_t>(mOps.size() / kOpFields);
    TLLM_CHECK_WITH_INFO(numTemplates >= 0 && mTemplates.front() == 0 && mTemplates.back() == numRuns
            && std::is_sorted(mTemplates.begin(), mTemplates.end()),
        "the template offsets must rise from 0 to the %ld runs", static_cast<long>(numRuns));
    for (int64_t run = 0; run < numRuns; ++run)
    {
        auto const* fields = &mRuns[run * kRunFields];
        TLLM_CHECK_WITH_INFO(fields[0] >= 0 && fields[0] < mNumComponents && fields[1] >= 0 && fields[1] < numRequests
                && fields[2] >= 0 && fields[3] > 0 && fields[4] >= 0,
            "run %ld has component %ld, request %ld, blocks %ld+%ld and offset %ld", static_cast<long>(run),
            static_cast<long>(fields[0]), static_cast<long>(fields[1]), static_cast<long>(fields[2]),
            static_cast<long>(fields[3]), static_cast<long>(fields[4]));
    }
    for (int64_t op = 0; op < numOps; ++op)
    {
        auto const* fields = &mOps[op * kOpFields];
        TLLM_CHECK_WITH_INFO((fields[0] == kSend || fields[0] == kReceive || fields[0] == kLocal) && fields[1] >= 0
                && fields[1] < numTemplates && fields[3] >= 0 && fields[4] > 0,
            "op %ld has action %ld, template %ld, offset %ld and %ld bytes", static_cast<long>(op),
            static_cast<long>(fields[0]), static_cast<long>(fields[1]), static_cast<long>(fields[3]),
            static_cast<long>(fields[4]));
    }

    auto const numSteps = mSteps.size() / kStepFields;
    std::fill(mStepBegin.begin(), mStepBegin.end(), 0);
    int64_t previous = -1;
    for (size_t step = 0; step < numSteps; ++step)
    {
        auto const* fields = &mSteps[step * kStepFields];
        auto const issuePoint = fields[0];
        TLLM_CHECK_WITH_INFO(issuePoint >= previous && issuePoint <= mNumLayers,
            "step %zu has issue point %ld after %ld; the steps must be in issue order up to %ld", step,
            static_cast<long>(issuePoint), static_cast<long>(previous), static_cast<long>(mNumLayers));
        TLLM_CHECK_WITH_INFO(
            fields[1] == 0 || fields[1] == 1, "step %zu has direction %ld", step, static_cast<long>(fields[1]));
        TLLM_CHECK_WITH_INFO(
            fields[2] >= 0 && fields[2] < mNumLayers, "step %zu is of layer %ld", step, static_cast<long>(fields[2]));
        TLLM_CHECK_WITH_INFO(fields[4] >= 0 && fields[5] >= 0 && fields[4] + fields[5] <= numOps,
            "step %zu has ops %ld+%ld of %ld", step, static_cast<long>(fields[4]), static_cast<long>(fields[5]),
            static_cast<long>(numOps));
        previous = issuePoint;
        mStepBegin[issuePoint + 2] = step + 1;
    }
    // A point without steps starts where the one before it ends.
    for (size_t point = 1; point < mStepBegin.size(); ++point)
    {
        mStepBegin[point] = std::max(mStepBegin[point], mStepBegin[point - 1]);
    }
    std::fill(mFetchRecorded.begin(), mFetchRecorded.end(), 0);
}

void DkvDataPlaneOp::run(int64_t issuePoint, int64_t computeStream, bool timeFetchWaits)
{
    TLLM_CHECK_WITH_INFO(issuePoint >= -1 && issuePoint <= mNumLayers, "issue point %ld is not in -1..%ld",
        static_cast<long>(issuePoint), static_cast<long>(mNumLayers));
    auto const compute = reinterpret_cast<cudaStream_t>(computeStream);
    auto const point = static_cast<size_t>(issuePoint + 1);
    if (issuePoint != 0)
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mComputeEvents[point], compute));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(mDataStream, mComputeEvents[point], 0));
    }
    for (size_t step = mStepBegin[point]; step < mStepBegin[point + 1]; ++step)
    {
        issueStep(step);
    }
    if (issuePoint >= 0 && issuePoint < mNumLayers && mFetchRecorded[issuePoint])
    {
        auto const layer = static_cast<size_t>(issuePoint);
        if (timeFetchWaits)
        {
            if (mWaitStarts.empty())
            {
                mWaitStarts = createEvents(mNumLayers, cudaEventDefault);
                mWaitEnds = createEvents(mNumLayers, cudaEventDefault);
            }
            TLLM_CUDA_CHECK(cudaEventRecord(mWaitStarts[layer], compute));
            TLLM_CUDA_CHECK(cudaStreamWaitEvent(compute, mFetchEvents[layer], 0));
            TLLM_CUDA_CHECK(cudaEventRecord(mWaitEnds[layer], compute));
            mWaitTimed[layer] = 1;
        }
        else
        {
            TLLM_CUDA_CHECK(cudaStreamWaitEvent(compute, mFetchEvents[layer], 0));
        }
    }
}

double DkvDataPlaneOp::fetchWaitMilliseconds()
{
    double total = 0.0;
    for (size_t layer = 0; layer < mWaitTimed.size(); ++layer)
    {
        if (!mWaitTimed[layer])
        {
            continue;
        }
        TLLM_CUDA_CHECK(cudaEventSynchronize(mWaitEnds[layer]));
        float milliseconds = 0.0F;
        TLLM_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, mWaitStarts[layer], mWaitEnds[layer]));
        total += milliseconds;
        mWaitTimed[layer] = 0;
    }
    return total;
}

void DkvDataPlaneOp::issueStep(size_t index)
{
    auto const* step = &mSteps[index * kStepFields];
    bool const fetch = step[1] == 0;
    auto const layer = step[2];
    auto const firstOp = step[4];
    auto const endOp = firstOp + step[5];
    auto const runsOf = [this](int64_t const* op)
    {
        auto const templateIndex = op[1];
        int64_t const* runs = mRuns.data();
        return std::make_pair(
            runs + mTemplates[templateIndex] * kRunFields, runs + mTemplates[templateIndex + 1] * kRunFields);
    };

    for (auto op = firstOp; op < endOp; ++op)
    {
        auto const* fields = &mOps[op * kOpFields];
        if (fields[0] != kLocal)
        {
            continue;
        }
        auto const [begin, end] = runsOf(fields);
        for (auto const* run = begin; run != end; run += kRunFields)
        {
            auto const pool = poolSide(layer, run);
            auto const staging = stagingSide(layer, run);
            addCopies(run, layer, fetch ? staging : pool, fetch ? pool : staging);
        }
    }
    launchCopies(fetch ? "the local fetch" : "the local writeback", layer);

    mSends.clear();
    for (auto op = firstOp; op < endOp; ++op)
    {
        auto const* fields = &mOps[op * kOpFields];
        if (fields[0] != kSend)
        {
            continue;
        }
        auto const base = mSendArena + fields[3];
        auto const [begin, end] = runsOf(fields);
        for (auto const* run = begin; run != end; run += kRunFields)
        {
            addCopies(run, layer, messageSide(base, run), fetch ? poolSide(layer, run) : stagingSide(layer, run));
        }
        mSends.push_back({base, fields[4], static_cast<int>(fields[2])});
    }
    launchCopies("packing a message", layer);
    if (!mSends.empty())
    {
        exchange(mSends, {});
    }

    mReceives.clear();
    for (auto op = firstOp; op < endOp; ++op)
    {
        auto const* fields = &mOps[op * kOpFields];
        if (fields[0] == kReceive)
        {
            mReceives.push_back({mRecvArena + fields[3], fields[4], static_cast<int>(fields[2])});
        }
    }
    if (!mReceives.empty())
    {
        exchange({}, mReceives);
        for (auto op = firstOp; op < endOp; ++op)
        {
            auto const* fields = &mOps[op * kOpFields];
            if (fields[0] != kReceive)
            {
                continue;
            }
            auto const base = mRecvArena + fields[3];
            auto const [begin, end] = runsOf(fields);
            for (auto const* run = begin; run != end; run += kRunFields)
            {
                addCopies(run, layer, fetch ? stagingSide(layer, run) : poolSide(layer, run), messageSide(base, run));
            }
        }
        launchCopies("unpacking a message", layer);
    }

    if (step[3] != 0)
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mFetchEvents[layer], mDataStream));
        mFetchRecorded[layer] = 1;
    }
}

DkvDataPlaneOp::Side DkvDataPlaneOp::poolSide(int64_t layer, int64_t const* run) const
{
    auto const* component = &mComponents[run[0] * kComponentFields];
    auto const* affine = &mPoolAffine[(layer * mNumAttentionTypes + component[1]) * kAffineFields];
    auto const group = affine[0];
    TLLM_CHECK_WITH_INFO(group >= 0 && group < mNumGroups,
        "rank %d: layer %ld, attention type %ld is not in the cache manager of this rank", mRank,
        static_cast<long>(layer), static_cast<long>(component[1]));
    auto const* request = &mRequests[run[1] * mRequestWidth];
    auto const address = request[1 + 2 * group];
    auto const length = request[2 + 2 * group];
    TLLM_CHECK_WITH_INFO(address != 0 && run[2] + run[3] <= length,
        "rank %d: the cache manager has no page for block %ld of request %ld in layer %ld (attention type %ld)", mRank,
        static_cast<long>(std::max<int64_t>(length, run[2])), static_cast<long>(request[0]), static_cast<long>(layer),
        static_cast<long>(component[1]));
    return {affine[2], affine[1], reinterpret_cast<int32_t const*>(address) + run[2]};
}

DkvDataPlaneOp::Side DkvDataPlaneOp::stagingSide(int64_t layer, int64_t const* run) const
{
    auto const* component = &mComponents[run[0] * kComponentFields];
    auto const kind = component[0];
    auto const pageBytes = component[2];
    auto const* request = &mRequests[run[1] * mRequestWidth];
    auto const* span = &request[1 + 2 * mNumGroups + 3 * kind];
    auto const firstBlock = span[0];
    auto const pageOffset = span[1];
    auto const numPages = span[2];
    TLLM_CHECK_WITH_INFO(
        numPages >= 0, "rank %d: request %ld is not in the staged batch", mRank, static_cast<long>(request[0]));
    TLLM_CHECK_WITH_INFO(firstBlock <= run[2] && run[2] + run[3] <= firstBlock + numPages,
        "rank %d: blocks %ld..%ld of request %ld are outside the staged blocks %ld..%ld of kind %ld", mRank,
        static_cast<long>(run[2]), static_cast<long>(run[2] + run[3] - 1), static_cast<long>(request[0]),
        static_cast<long>(firstBlock), static_cast<long>(firstBlock + numPages - 1), static_cast<long>(kind));
    auto const pointer = mStagingPointers[layer * mNumComponents + run[0]];
    TLLM_CHECK_WITH_INFO(pointer != 0, "rank %d: layer %ld has no staging slot for component %ld", mRank,
        static_cast<long>(layer), static_cast<long>(run[0]));
    auto const entry = mSlotOffsets[layer * mNumKinds + kind] + pageOffset + run[2] - firstBlock;
    return {pointer + entry * pageBytes, pageBytes, nullptr};
}

DkvDataPlaneOp::Side DkvDataPlaneOp::messageSide(int64_t base, int64_t const* run) const
{
    auto const pageBytes = mComponents[run[0] * kComponentFields + 2];
    return {base + run[4], pageBytes, nullptr};
}

void DkvDataPlaneOp::addCopies(int64_t const* run, int64_t layer, Side const& destination, Side const& source)
{
    auto const pageBytes = mComponents[run[0] * kComponentFields + 2];
    auto entry = std::find_if(
        mCopies.begin(), mCopies.end(), [pageBytes](auto const& copies) { return copies.first == pageBytes; });
    if (entry == mCopies.end())
    {
        mCopies.emplace_back(pageBytes, std::vector<CopyTask>{});
        entry = std::prev(mCopies.end());
    }
    auto& tasks = entry->second;
    for (int64_t page = 0; page < run[3]; ++page)
    {
        int64_t const to = destination.indices != nullptr ? destination.indices[page] : page;
        int64_t const from = source.indices != nullptr ? source.indices[page] : page;
        TLLM_CHECK_WITH_INFO(to >= 0 && from >= 0,
            "rank %d: block %ld of request %ld in layer %ld has no page in the cache manager", mRank,
            static_cast<long>(run[2] + page), static_cast<long>(requestId(run[1])), static_cast<long>(layer));
        tasks.push_back(CopyTask{static_cast<kvc::MemAddress>(destination.base + to * destination.step),
            static_cast<kvc::MemAddress>(source.base + from * source.step)});
    }
}

void DkvDataPlaneOp::launchCopies(char const* what, int64_t layer)
{
    for (auto& [pageBytes, tasks] : mCopies)
    {
        if (tasks.empty())
        {
            continue;
        }
        auto const result = kvc::copyDeviceToDevice(tasks, pageBytes, reinterpret_cast<CUstream>(mDataStream));
        tasks.clear();
        TLLM_CHECK_WITH_INFO(result == CUDA_SUCCESS, "rank %d: %s of layer %ld failed to launch (CUDA error %d)", mRank,
            what, static_cast<long>(layer), static_cast<int>(result));
    }
}

void DkvDataPlaneOp::exchange(std::vector<Message> const& sends, std::vector<Message> const& receives) const
{
#if ENABLE_MULTI_DEVICE
    tr::CudaStream stream{mDataStream, mDevice, false};
    auto bytesAt = [](int64_t address, int64_t size)
    { return tr::IBuffer::wrap(reinterpret_cast<std::uint8_t*>(static_cast<uintptr_t>(address)), size); };
    TLLM_NCCL_CHECK(ncclGroupStart());
    for (auto const& message : sends)
    {
        mComm->send(*bytesAt(message.address, message.bytes), message.peer, stream);
    }
    for (auto const& message : receives)
    {
        mComm->receive(*bytesAt(message.address, message.bytes), message.peer, stream);
    }
    TLLM_NCCL_CHECK(ncclGroupEnd());
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

int64_t DkvDataPlaneOp::requestId(int64_t request) const
{
    return mRequests[request * mRequestWidth];
}

th::jit::class_<DkvDataPlaneOp> registerDkvDataPlane()
{
    return th::jit::class_<DkvDataPlaneOp>("trtllm", "DkvDataPlane")
        .def(th::jit::init<c10::intrusive_ptr<NcclCommunicatorOp>, int64_t, int64_t>())
        .def("set_static", &DkvDataPlaneOp::setStatic)
        .def("set_iteration", &DkvDataPlaneOp::setIteration)
        .def("run", &DkvDataPlaneOp::run)
        .def("fetch_wait_ms", &DkvDataPlaneOp::fetchWaitMilliseconds);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
