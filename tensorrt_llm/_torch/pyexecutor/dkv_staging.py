# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DKV layer split: copies of KV pages between caches and message buffers, on a stream.

A copy is a list of *runs*: the pages one buffer holds for one request, each side given by a base
address, an address step and, when the pages do not follow each other, the page indices of the
blocks. ``DkvPageCopier`` enqueues the copies of runs with the copy kernel of the V2 cache manager.
"""

import ctypes
from collections.abc import Sequence

import numpy as np
import torch

# A run of pages of ``DkvPageCopier.copy_runs`` is this many integers: the page bytes, the page
# count, then for the destination and for the source a base address, an address step and the host
# address of ``count`` int32 page indices, or 0 when the pages follow each other.
RUN_FIELDS = 8


def run_addresses(base: int, step: int, indices: int, count: int) -> np.ndarray:
    """The addresses of the pages of one side of a run of ``copy_runs``.

    Raises:
        RuntimeError: A page index is negative, which marks a block without a page.
    """
    if not indices:
        return base + np.arange(count, dtype=np.int64) * step
    table = np.ctypeslib.as_array(
        ctypes.cast(indices, ctypes.POINTER(ctypes.c_int32)), shape=(count,)
    ).astype(np.int64)
    if (table < 0).any():
        raise RuntimeError("a run names a block without a page")
    return base + table * step


def expand_runs(runs: Sequence[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """The ``(page bytes, destination addresses, source addresses)`` of runs, one per page size."""
    if len(runs) % RUN_FIELDS:
        raise ValueError(f"a run is {RUN_FIELDS} integers, {len(runs)} were given")
    by_size: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for start in range(0, len(runs), RUN_FIELDS):
        page_bytes, count, dst_base, dst_step, dst_indices, src_base, src_step, src_indices = runs[
            start : start + RUN_FIELDS
        ]
        group = by_size.setdefault(page_bytes, ([], []))
        group[0].append(run_addresses(dst_base, dst_step, dst_indices, count))
        group[1].append(run_addresses(src_base, src_step, src_indices, count))
    return [
        (page_bytes, np.concatenate(destinations), np.concatenate(sources))
        for page_bytes, (destinations, sources) in by_size.items()
    ]


class DkvPageCopier:
    """Copies the pages of runs between caches and message buffers, on a stream."""

    # The copy kernel receives its tasks by value in its launch arguments.
    MAX_TASKS_PER_CALL = 256

    def __init__(self) -> None:
        from tensorrt_llm.bindings.internal.batch_manager import kv_cache_manager_v2_utils

        self._utils = kv_cache_manager_v2_utils
        # A native build before the address-tensor entry point takes one task object per page, and
        # one before the run entry point expands the runs here.
        self._copy_addresses = getattr(
            kv_cache_manager_v2_utils, "copy_device_to_device_addresses", None
        )
        self._copy_runs = getattr(kv_cache_manager_v2_utils, "copy_device_to_device_runs", None)

    def copy_runs(self, runs: Sequence[int], stream: int) -> None:
        """Copy the pages of runs, ``RUN_FIELDS`` integers each (see ``RUN_FIELDS``).

        Page ``i`` of a side is at ``base + indices[i] * step``, or at ``base + i * step`` when the
        side has no index address. The index tables are read now, on the host, so they only have
        to live until the call returns.

        Raises:
            RuntimeError: A block has no page, or the copy could not be enqueued.
        """
        if len(runs) % RUN_FIELDS:
            raise ValueError(f"a run is {RUN_FIELDS} integers, {len(runs)} were given")
        if self._copy_runs is None:
            for page_bytes, destinations, sources in expand_runs(runs):
                self.copy_addresses(destinations, sources, page_bytes, stream)
            return
        result = self._copy_runs(runs if isinstance(runs, list) else list(runs), stream)
        if result == -1:
            raise RuntimeError("a run names a block without a page")
        if result != 0:
            raise RuntimeError(f"copy_device_to_device_runs failed with CUDA error {result}")

    def copy_addresses(
        self, destinations: np.ndarray, sources: np.ndarray, num_bytes: int, stream: int
    ) -> None:
        """Copy ``num_bytes`` from every source address to the destination address at its position.

        Args:
            destinations: The int64 device addresses the copies write to, one per page.
            sources: The int64 device addresses the copies read from, as many as ``destinations``.
            num_bytes: The size of every copy, a positive multiple of 16.
            stream: The CUDA stream handle the copies are enqueued on.
        """
        if num_bytes <= 0 or num_bytes % 16:
            raise ValueError(f"num_bytes must be a positive multiple of 16, got {num_bytes}")
        if len(destinations) != len(sources):
            raise ValueError(
                f"{len(destinations)} destination addresses but {len(sources)} source addresses"
            )
        if self._copy_addresses is None:
            self.copy(list(zip(destinations.tolist(), sources.tolist())), num_bytes, stream)
            return
        result = self._copy_addresses(
            torch.from_numpy(np.ascontiguousarray(destinations, dtype=np.int64)),
            torch.from_numpy(np.ascontiguousarray(sources, dtype=np.int64)),
            num_bytes,
            stream,
        )
        if result != 0:
            raise RuntimeError(f"copy_device_to_device_addresses failed with CUDA error {result}")

    def copy(self, pairs: Sequence[tuple[int, int]], num_bytes: int, stream: int) -> None:
        """Copy ``num_bytes`` from the source to the destination address of every pair.

        Args:
            pairs: ``(destination address, source address)`` of device memory.
            num_bytes: The size of every copy, a positive multiple of 16.
            stream: The CUDA stream handle the copies are enqueued on.
        """
        if num_bytes <= 0 or num_bytes % 16:
            raise ValueError(f"num_bytes must be a positive multiple of 16, got {num_bytes}")
        for begin in range(0, len(pairs), self.MAX_TASKS_PER_CALL):
            tasks = [
                self._utils.MemToMemTask(destination, source)
                for destination, source in pairs[begin : begin + self.MAX_TASKS_PER_CALL]
            ]
            result = self._utils.copy_device_to_device(tasks, num_bytes, stream)
            if result != 0:
                raise RuntimeError(f"copy_device_to_device failed with CUDA error {result}")
