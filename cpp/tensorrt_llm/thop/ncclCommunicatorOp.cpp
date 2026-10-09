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

#include "tensorrt_llm/thop/ncclCommunicatorOp.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <c10/cuda/CUDAStream.h>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

namespace tr = tensorrt_llm::runtime;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

//! The bytes of a tensor as a buffer the communicator can send or receive.
tr::IBuffer::UniquePtr bytesOf(th::Tensor const& tensor)
{
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    return tr::IBuffer::wrap(ptr, size);
}

//! Keep the memory of `tensor` alive for the work queued on the stream of `handle`.
void recordStream(th::Tensor const& tensor, int64_t handle)
{
    tensor.record_stream(c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(handle), tensor.get_device()));
}

} // namespace

NcclCommunicatorOp::NcclCommunicatorOp(int64_t worldSize, int64_t rank)
    : mRank(static_cast<int32_t>(rank))
{
    mPipelineComm = std::make_shared<tensorrt_llm::runtime::NcclCommunicator>(worldSize, rank);
}

void NcclCommunicatorOp::sendOn(th::Tensor tensor, int64_t toRank, int64_t stream) const
{
    recordStream(tensor, stream);
    tensorrt_llm::runtime::CudaStream cudaStream{reinterpret_cast<cudaStream_t>(stream), mRank, false};
    mPipelineComm->send(*bytesOf(tensor), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recvOn(th::Tensor& tensor, int64_t fromRank, int64_t stream) const
{
    recordStream(tensor, stream);
    tensorrt_llm::runtime::CudaStream cudaStream{reinterpret_cast<cudaStream_t>(stream), mRank, false};
    mPipelineComm->receive(*bytesOf(tensor), static_cast<int>(fromRank), cudaStream);
}

void NcclCommunicatorOp::groupSendRecv(std::vector<th::Tensor> sendTensors, std::vector<int64_t> sendPeers,
    std::vector<th::Tensor> recvTensors, std::vector<int64_t> recvPeers, int64_t stream) const
{
#if ENABLE_MULTI_DEVICE
    TLLM_CHECK_WITH_INFO(sendTensors.size() == sendPeers.size(),
        "Every send tensor needs a peer: %zu tensors, %zu peers", sendTensors.size(), sendPeers.size());
    TLLM_CHECK_WITH_INFO(recvTensors.size() == recvPeers.size(),
        "Every receive tensor needs a peer: %zu tensors, %zu peers", recvTensors.size(), recvPeers.size());
    tensorrt_llm::runtime::CudaStream cudaStream{reinterpret_cast<cudaStream_t>(stream), mRank, false};
    for (auto const& tensor : sendTensors)
    {
        recordStream(tensor, stream);
    }
    for (auto const& tensor : recvTensors)
    {
        recordStream(tensor, stream);
    }
    TLLM_NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < sendTensors.size(); ++i)
    {
        mPipelineComm->send(*bytesOf(sendTensors[i]), static_cast<int>(sendPeers[i]), cudaStream);
    }
    for (size_t i = 0; i < recvTensors.size(); ++i)
    {
        mPipelineComm->receive(*bytesOf(recvTensors[i]), static_cast<int>(recvPeers[i]), cudaStream);
    }
    TLLM_NCCL_CHECK(ncclGroupEnd());
#else
    TLLM_THROW("Multi device support is disabled.");
#endif // ENABLE_MULTI_DEVICE
}

void NcclCommunicatorOp::send(th::Tensor tensor, int64_t toRank) const
{
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->send(*tr::IBuffer::wrap(ptr, size), static_cast<int>(toRank), cudaStream);
}

void NcclCommunicatorOp::recv(th::Tensor& tensor, int64_t fromRank) const
{
    tensor.record_stream(at::cuda::getCurrentCUDAStream());
    auto ptr = static_cast<std::uint8_t*>(tensor.data_ptr());
    size_t const size = tensor.numel() * th::elementSize(th::typeMetaToScalarType(tensor.dtype()));
    tensorrt_llm::runtime::CudaStream cudaStream{at::cuda::getCurrentCUDAStream().stream(), mRank, false};
    mPipelineComm->receive(*tr::IBuffer::wrap(ptr, size), static_cast<int>(fromRank), cudaStream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

static auto trtllmNcclCommunicator
    = torch::jit::class_<tensorrt_llm::torch_ext::NcclCommunicatorOp>("trtllm", "NcclCommunicatorOp")
          .def(torch::jit::init<int64_t, int64_t>())
          .def("send", &tensorrt_llm::torch_ext::NcclCommunicatorOp::send)
          .def("recv", &tensorrt_llm::torch_ext::NcclCommunicatorOp::recv)
          .def("send_on", &tensorrt_llm::torch_ext::NcclCommunicatorOp::sendOn)
          .def("recv_on", &tensorrt_llm::torch_ext::NcclCommunicatorOp::recvOn)
          .def("group_send_recv", &tensorrt_llm::torch_ext::NcclCommunicatorOp::groupSendRecv);
