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

"""Shared bounded thread runners and in-process distributed test helpers."""

import threading
import time
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


class ThreadSafeDistributed:
    """Distributed mock using threading.Barrier for single-process multi-rank testing."""

    def __init__(
        self,
        local_rank: int,
        world_size: int,
        tp_size: int,
        pp_size: int,
        tp_rank: int,
        pp_rank: int,
        shared: dict,
        cp_rank: int = 0,
        cp_size: int = 1,
        barrier_timeout: float | None = None,
    ) -> None:
        self.rank = local_rank
        self._world_size = world_size
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._tp_rank = tp_rank
        self._pp_rank = pp_rank
        # TP/PP groups exchange independently within each CP slice.
        self._cp_rank = cp_rank
        self._cp_size = cp_size
        self._s = shared
        # None waits indefinitely; a bound turns a stuck peer into BrokenBarrierError.
        self._barrier_timeout = barrier_timeout
        self._bcast_idx = 0
        self._ag_idx = 0
        self._pp_ag_idx = 0
        self._tp_ag_idx = 0

    @property
    def tp_size(self) -> int:
        return self._tp_size

    @property
    def pp_size(self) -> int:
        return self._pp_size

    @property
    def world_size(self) -> int:
        return self._world_size

    def broadcast(self, obj: object, root: int = 0) -> object:
        idx = self._bcast_idx
        self._bcast_idx += 1
        key = f"bcast_{idx}"
        if self.rank == root:
            self._s[key] = obj
        self._s["barrier"].wait(timeout=self._barrier_timeout)
        result = self._s[key]
        self._s["barrier"].wait(timeout=self._barrier_timeout)
        return result

    def allgather(self, obj: object) -> list[object]:
        idx = self._ag_idx
        self._ag_idx += 1
        key = f"ag_{idx}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._world_size
            self._s[key][self.rank] = obj
        self._s["barrier"].wait(timeout=self._barrier_timeout)
        result = list(self._s[key])
        self._s["barrier"].wait(timeout=self._barrier_timeout)
        return result

    def pp_allgather(self, obj: object) -> list[object]:
        idx = self._pp_ag_idx
        self._pp_ag_idx += 1
        key = f"pp_ag_{idx}_tp{self._tp_rank}_cp{self._cp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._pp_size
            self._s[key][self._pp_rank] = obj
        # Sync only the PP group that shares this (tp_rank, cp_rank). With attention data
        # parallelism each tp_rank is an independent instance that may run a different number
        # of collectives, so a single global barrier would deadlock; a per-group one does not.
        # CP adds an orthogonal axis, so the group is keyed by cp_rank too.
        pp_barrier = (
            self._s["pp_barriers"][self._tp_rank * self._cp_size + self._cp_rank]
            if "pp_barriers" in self._s
            else self._s["barrier"]
        )
        pp_barrier.wait(timeout=self._barrier_timeout)
        result = list(self._s[key])
        pp_barrier.wait(timeout=self._barrier_timeout)
        return result

    def tp_allgather(self, obj: object) -> list[object]:
        idx = self._tp_ag_idx
        self._tp_ag_idx += 1
        key = f"tp_ag_{idx}_pp{self._pp_rank}_cp{self._cp_rank}"
        with self._s["lock"]:
            if key not in self._s:
                self._s[key] = [None] * self._tp_size
            self._s[key][self._tp_rank] = obj
        # Sync only the TP group that shares this (pp_rank, cp_rank) (see pp_allgather).
        tp_barrier = (
            self._s["tp_barriers"][self._pp_rank * self._cp_size + self._cp_rank]
            if "tp_barriers" in self._s
            else self._s["barrier"]
        )
        tp_barrier.wait(timeout=self._barrier_timeout)
        result = list(self._s[key])
        tp_barrier.wait(timeout=self._barrier_timeout)
        return result


def run_concurrent(
    items: Sequence[_Item],
    fn: Callable[[_Item], _Result],
    *,
    timeout: float | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> list[_Result]:
    """Run each item on a thread and re-raise the first worker error on the calling thread.

    Without ``timeout`` the threads are joined without a bound. With one, the threads are daemons
    and every join is bounded; ``on_error`` should then abort any collective on which a peer may be
    waiting, so a failing native callback cannot hang pytest exit.
    """
    errors: list[Exception | None] = [None] * len(items)
    results: list[_Result | None] = [None] * len(items)

    def worker(index: int, item: _Item) -> None:
        try:
            results[index] = fn(item)
        except Exception as error:
            # Worker failures must be re-raised on the calling test thread.
            errors[index] = error
            if on_error is not None:
                on_error(error)

    threads = [
        threading.Thread(target=worker, args=(index, item), daemon=timeout is not None)
        for index, item in enumerate(items)
    ]
    for thread in threads:
        thread.start()
    if timeout is None:
        for thread in threads:
            thread.join()
    else:
        deadline = time.monotonic() + timeout
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        alive = [index for index, thread in enumerate(threads) if thread.is_alive()]
        if alive:
            error = TimeoutError(f"Concurrent workers did not finish: {alive}")
            if on_error is not None:
                on_error(error)
            for thread in threads:
                thread.join(timeout=1)
            raise error
    for error in errors:
        if error is not None:
            raise error
    return cast(list[_Result], results)
