/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kvCacheManagerV2Utils.h"
#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchView.h"
#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <torch/extension.h>

#include <algorithm>
#include <utility>

namespace tr = tensorrt_llm::runtime;
namespace nb = nanobind;

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

std::optional<tensorrt_llm::runtime::ITensor::UniquePtr> from_torch(std::optional<at::Tensor> torchPtr)
{
    if (torchPtr)
    {
        return tr::TorchView::of(torchPtr.value());
    }
    return std::nullopt;
}

void KVCacheManagerV2UtilsBindings::initBindings(nb::module_& module)
{
    // Bind DiskAddress struct
    nb::class_<DiskAddress>(module, "DiskAddress")
        .def(nb::init<int, ssize_t>(), nb::arg("fd"), nb::arg("pos"))
        .def_rw("fd", &DiskAddress::fd)
        .def_rw("pos", &DiskAddress::pos);

    // Bind Task template instantiations
    nb::class_<Task<DiskAddress, DiskAddress>>(module, "DiskToDiskTask")
        .def(nb::init<DiskAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, DiskAddress>::dst)
        .def_rw("src", &Task<DiskAddress, DiskAddress>::src);

    nb::class_<Task<MemAddress, DiskAddress>>(module, "DiskToHostTask")
        .def(nb::init<MemAddress, DiskAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, DiskAddress>::dst)
        .def_rw("src", &Task<MemAddress, DiskAddress>::src);

    nb::class_<Task<DiskAddress, MemAddress>>(module, "HostToDiskTask")
        .def(nb::init<DiskAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<DiskAddress, MemAddress>::dst)
        .def_rw("src", &Task<DiskAddress, MemAddress>::src);

    nb::class_<Task<MemAddress, MemAddress>>(module, "MemToMemTask")
        .def(nb::init<MemAddress, MemAddress>(), nb::arg("dst"), nb::arg("src"))
        .def_rw("dst", &Task<MemAddress, MemAddress>::dst)
        .def_rw("src", &Task<MemAddress, MemAddress>::src);

    nb::class_<IndexMapper>(module, "IndexMapper")
        .def(nb::init<SizeType32, SizeType32>(), nb::arg("max_batch_size"), nb::arg("max_beam_width"))
        .def("add_new_sequence", &IndexMapper::addNewSequence)
        .def("get_index", &IndexMapper::getIndex)
        .def("remove_sequence", &IndexMapper::removeSequence)
        .def("get_copy_index", &IndexMapper::getCopyIndex)
        .def("gather_k_block_offsets", &IndexMapper::gatherKBlockOffsets, nb::arg("source"), nb::arg("destination"),
            nb::arg("request_ids"), nb::arg("num_blocks"))
        .def("size", &IndexMapper::size)
        .def("num_free_slots", &IndexMapper::numFreeSlots);

    // Bind copy functions
    module.def(
        "copy_disk_to_disk",
        [](std::vector<Task<DiskAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to disk using CUDA host function");

    module.def(
        "copy_disk_to_host",
        [](std::vector<Task<MemAddress, DiskAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDiskToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from disk to host using CUDA host function");

    module.def(
        "copy_host_to_disk",
        [](std::vector<Task<DiskAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDisk(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to disk using CUDA host function");

    module.def(
        "copy_host_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToHost(std::move(tasks), numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to host using CUDA host function");

    module.def(
        "copy_host_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyHostToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from host to device using CUDA kernels");

    module.def(
        "copy_device_to_host",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToHost(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to host using CUDA kernels");

    module.def(
        "copy_device_to_device",
        [](std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, uintptr_t stream) -> int
        { return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream)); },
        nb::arg("tasks"), nb::arg("num_bytes"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy data from device to device using CUDA kernels");

    module.def(
        "copy_device_to_device_addresses",
        [](at::Tensor const& destinations, at::Tensor const& sources, ssize_t numBytes, uintptr_t stream) -> int
        {
            TLLM_CHECK_WITH_INFO(destinations.device().is_cpu() && sources.device().is_cpu(),
                "The address tensors must live in host memory.");
            TLLM_CHECK_WITH_INFO(destinations.scalar_type() == at::kLong && sources.scalar_type() == at::kLong,
                "The address tensors must hold int64 device addresses.");
            TLLM_CHECK_WITH_INFO(
                destinations.is_contiguous() && sources.is_contiguous(), "The address tensors must be contiguous.");
            TLLM_CHECK_WITH_INFO(destinations.numel() == sources.numel(),
                "The address tensors must have the same length: %ld destinations, %ld sources.",
                static_cast<long>(destinations.numel()), static_cast<long>(sources.numel()));
            auto const count = static_cast<size_t>(destinations.numel());
            auto const* dst = destinations.data_ptr<int64_t>();
            auto const* src = sources.data_ptr<int64_t>();
            std::vector<Task<MemAddress, MemAddress>> tasks(count);
            for (size_t i = 0; i < count; ++i)
            {
                tasks[i]
                    = Task<MemAddress, MemAddress>{static_cast<MemAddress>(dst[i]), static_cast<MemAddress>(src[i])};
            }
            return copyDeviceToDevice(tasks, numBytes, reinterpret_cast<CUstream>(stream));
        },
        nb::arg("destinations"), nb::arg("sources"), nb::arg("num_bytes"), nb::arg("stream"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Copy num_bytes of device memory from every source address to the destination address at the same "
        "position; the addresses are int64 host tensors");

    // A run is eight integers: the page bytes, the page count, then for the destination and for the source a
    // base address, an address step and the host address of `count` int32 page indices (0: the pages follow
    // each other). Page i of a side is at base + (indices ? indices[i] : i) * step. A negative index is a block
    // without a page.
    module.def(
        "copy_device_to_device_runs",
        [](std::vector<int64_t> const& runs, uintptr_t stream) -> int
        {
            constexpr size_t kFields = 8;
            TLLM_CHECK_WITH_INFO(
                runs.size() % kFields == 0, "A run is %zu integers, %zu were given.", kFields, runs.size());
            std::vector<std::pair<int64_t, std::vector<Task<MemAddress, MemAddress>>>> bySize;
            for (size_t r = 0; r < runs.size(); r += kFields)
            {
                int64_t const pageBytes = runs[r];
                int64_t const count = runs[r + 1];
                int64_t const dstBase = runs[r + 2];
                int64_t const dstStep = runs[r + 3];
                auto const* dstIndices = reinterpret_cast<int32_t const*>(static_cast<uintptr_t>(runs[r + 4]));
                int64_t const srcBase = runs[r + 5];
                int64_t const srcStep = runs[r + 6];
                auto const* srcIndices = reinterpret_cast<int32_t const*>(static_cast<uintptr_t>(runs[r + 7]));
                auto group = std::find_if(
                    bySize.begin(), bySize.end(), [pageBytes](auto const& entry) { return entry.first == pageBytes; });
                if (group == bySize.end())
                {
                    bySize.emplace_back(pageBytes, std::vector<Task<MemAddress, MemAddress>>{});
                    group = std::prev(bySize.end());
                }
                auto& tasks = group->second;
                tasks.reserve(tasks.size() + static_cast<size_t>(std::max<int64_t>(count, 0)));
                for (int64_t i = 0; i < count; ++i)
                {
                    int64_t const d = dstIndices != nullptr ? dstIndices[i] : i;
                    int64_t const s = srcIndices != nullptr ? srcIndices[i] : i;
                    if (d < 0 || s < 0)
                    {
                        return -1;
                    }
                    tasks.push_back(Task<MemAddress, MemAddress>{static_cast<MemAddress>(dstBase + d * dstStep),
                        static_cast<MemAddress>(srcBase + s * srcStep)});
                }
            }
            for (auto const& [pageBytes, tasks] : bySize)
            {
                auto const result = copyDeviceToDevice(tasks, pageBytes, reinterpret_cast<CUstream>(stream));
                if (result != CUDA_SUCCESS)
                {
                    return static_cast<int>(result);
                }
            }
            return 0;
        },
        nb::arg("runs"), nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(),
        "Copy the pages of runs of device memory, eight integers per run (page bytes, count, destination base, "
        "step and index address, source base, step and index address); returns -1 for a block without a page, "
        "else the CUDA result");

    module.def(
        "copy_batch_block_offsets_to_device",
        [](at::Tensor input, at::Tensor output, at::Tensor copyIndex, at::Tensor indexScales, at::Tensor kvOffset,
            uintptr_t stream)
        {
            auto _input = from_torch(input);
            auto _output = from_torch(output);
            auto _copyIndex = from_torch(copyIndex);
            auto _indexScales = from_torch(indexScales);
            auto _kvOffset = from_torch(kvOffset);
            TLLM_CHECK_WITH_INFO(_input.has_value(), "Invalid input tensor.");
            TLLM_CHECK_WITH_INFO(_output.has_value(), "Invalid output tensor.");
            TLLM_CHECK_WITH_INFO(_copyIndex.has_value(), "Invalid copy index tensor.");
            TLLM_CHECK_WITH_INFO(_indexScales.has_value(), "Invalid index scales tensor.");
            TLLM_CHECK_WITH_INFO(_kvOffset.has_value(), "Invalid kv offset tensor.");
            copyBatchBlockOffsetsToDevice(*(_input.value()), *(_output.value()), *(_copyIndex.value()),
                *(_indexScales.value()), *(_kvOffset.value()), reinterpret_cast<CUstream>(stream));
        },
        nb::arg("input"), nb::arg("output"), nb::arg("copy_index"), nb::arg("index_scales"), nb::arg("kv_offset"),
        nb::arg("stream"), nb::call_guard<nb::gil_scoped_release>(), "Copy batch block indices to device");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
