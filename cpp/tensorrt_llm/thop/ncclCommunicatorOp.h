/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/ncclCommunicator.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <memory>

namespace th = torch;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

class NcclCommunicatorOp : public th::jit::CustomClassHolder
{
public:
    NcclCommunicatorOp(int64_t worldSize, int64_t rank);

    void send(th::Tensor tensor, int64_t toRank) const;
    void recv(th::Tensor& tensor, int64_t fromRank) const;

    //! Like send and recv, on the CUDA stream whose handle is `stream` instead of the current stream.
    void sendOn(th::Tensor tensor, int64_t toRank, int64_t stream) const;
    void recvOn(th::Tensor& tensor, int64_t fromRank, int64_t stream) const;

    //! Issue every send and receive of the lists inside one NCCL group on the stream whose handle is `stream`.
    //! NCCL runs the group as one kernel; the sends and receives to one peer pair up with the peer's in list
    //! order, as the individual calls would.
    void groupSendRecv(std::vector<th::Tensor> sendTensors, std::vector<int64_t> sendPeers,
        std::vector<th::Tensor> recvTensors, std::vector<int64_t> recvPeers, int64_t stream) const;

    //! groupSendRecv on raw device addresses and byte counts. The caller keeps the memory alive until the work
    //! queued on the stream is done.
    void groupSendRecvRaw(std::vector<int64_t> sendAddresses, std::vector<int64_t> sendBytes,
        std::vector<int64_t> sendPeers, std::vector<int64_t> recvAddresses, std::vector<int64_t> recvBytes,
        std::vector<int64_t> recvPeers, int64_t stream) const;

private:
    int32_t mRank;
    std::shared_ptr<tensorrt_llm::runtime::NcclCommunicator> mPipelineComm;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
