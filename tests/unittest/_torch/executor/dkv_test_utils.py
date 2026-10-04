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
"""Real requests and strict, bounded TP collectives for DKV control-flow tests."""

import copy
import inspect
import operator
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from typing import TypeVar

import numpy as np
from utils.collectives import run_concurrent

from tensorrt_llm import Mapping
from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

_Result = TypeVar("_Result")


def make_request(
    request_id: int,
    *,
    compute_rank: int = 0,
    local_rank: int = 0,
    prompt_len: int = 8,
    is_dummy: bool = False,
) -> LlmRequest:
    """Build a request with concrete DKV tags and no CUDA allocations."""
    request = LlmRequest(
        request_id=request_id,
        input_tokens=list(range(prompt_len)),
        max_new_tokens=1,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
    )
    request.py_dkv_compute_rank = compute_rank
    request.py_dkv_is_local = compute_rank == local_rank
    request.is_attention_dp_dummy = is_dummy
    return request


@dataclass
class _CollectiveStep:
    signature: tuple[object, ...]
    payloads: dict[int, object] = field(default_factory=dict)


class LockstepTpGroup:
    """Execute rank callbacks with real payload exchange and sequence checking.

    Every collective shares one sequence counter per rank. Calls must agree
    on method, source location and parameters. Returning early, skipping a
    collective or failing on a peer wakes the remaining ranks immediately.
    """

    def __init__(self, size: int, *, timeout: float = 5.0) -> None:
        if size < 1 or timeout <= 0:
            raise ValueError("size and timeout must be positive")
        self.size = size
        self.timeout = timeout
        self._condition = threading.Condition()
        self._steps: dict[int, _CollectiveStep] = {}
        self._finished: set[int] = set()
        self._error: Exception | None = None
        self.traces: list[list[tuple[object, ...]]] = [[] for _ in range(size)]
        self.ranks = [LockstepDistributed(self, rank) for rank in range(size)]

    def _diagnostic(self, message: str) -> AssertionError:
        traces = "\n".join(f"  rank{rank}: {trace!r}" for rank, trace in enumerate(self.traces))
        return AssertionError(f"{message}\n{traces}")

    def abort(self, error: Exception) -> None:
        """Unblock every rank after a callback or collective fails."""
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def _exchange(
        self, rank: int, sequence: int, signature: tuple[object, ...], payload: object
    ) -> list[object]:
        with self._condition:
            self.traces[rank].append((sequence, *signature))
            if self._error is not None:
                raise self._error
            if self._finished:
                error = self._diagnostic(
                    f"Collective {sequence} reached after ranks {sorted(self._finished)} finished"
                )
                self.abort(error)
                raise error
            step = self._steps.setdefault(sequence, _CollectiveStep(signature))
            if signature != step.signature:
                error = self._diagnostic(f"Collective sequence mismatch at {sequence}")
                self.abort(error)
                raise error
            step.payloads[rank] = copy.deepcopy(payload)
            self._condition.notify_all()
            deadline = time.monotonic() + self.timeout
            while len(step.payloads) < self.size and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.abort(self._diagnostic(f"Collective timeout at {sequence}"))
                    break
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise self._error
            return copy.deepcopy([step.payloads[index] for index in range(self.size)])

    def _finish(self, rank: int) -> None:
        with self._condition:
            self._finished.add(rank)
            if any(len(step.payloads) < self.size for step in self._steps.values()):
                self.abort(self._diagnostic(f"Rank {rank} finished with an unmatched collective"))

    def run(self, fn: Callable[["LockstepDistributed"], _Result]) -> list[_Result]:
        """Run rank callbacks and abort collective waits when a peer fails."""
        self._finished.clear()

        def worker(dist: LockstepDistributed) -> _Result:
            try:
                return fn(dist)
            except Exception as error:
                # Preserve the source failure before notifying waiting peers.
                self.abort(error)
                raise
            finally:
                self._finish(dist.tp_rank)

        return run_concurrent(self.ranks, worker, timeout=self.timeout + 2, on_error=self.abort)


class LockstepDistributed:
    """Small explicit Distributed substitute; unsupported collectives fail closed."""

    def __init__(self, group: LockstepTpGroup, rank: int) -> None:
        self._group = group
        self._sequence = 0
        self.rank = rank
        self.tp_rank = rank
        self.tp_size = group.size
        self.world_size = group.size
        self.pp_rank = 0
        self.pp_size = 1
        self.cp_rank = 0
        self.cp_size = 1
        self.has_cp_helix = False
        self.mapping = Mapping(
            world_size=group.size, rank=rank, tp_size=group.size, enable_attention_dp=True
        )
        self.cp_config = self.mapping.cp_config

    def __getattr__(self, name: str) -> object:
        if any(word in name for word in ("gather", "reduce", "broadcast", "barrier")) or (
            name in Distributed.__dict__ and callable(Distributed.__dict__[name])
        ):

            def unwired(*args: object, **kwargs: object) -> None:
                raise AssertionError(f"unwired collective: {name}")

            return unwired
        raise AttributeError(name)

    def _exchange(self, kind: str, payload: object, *parameters: object) -> list[object]:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        caller = frame.f_back.f_back
        assert caller is not None
        site = (caller.f_code.co_filename, caller.f_code.co_name, caller.f_lineno)
        del frame, caller
        sequence = self._sequence
        self._sequence += 1
        return self._group._exchange(self.tp_rank, sequence, (site, kind, *parameters), payload)

    def tp_allgather(self, obj: object, *, small_payload: bool = False) -> list[object]:
        return self._exchange("tp_allgather", obj, small_payload)

    def tp_allgather_int64(self, values: object) -> np.ndarray:
        vector = np.asarray(values, dtype=np.int64).reshape(-1)
        result = self._exchange("tp_allgather_int64", vector, vector.size)
        return np.asarray(result, dtype=np.int64)

    def tp_gather(self, obj: object, root: int = 0) -> list[object] | None:
        if not 0 <= root < self.tp_size:
            raise ValueError("root outside TP group")
        result = self._exchange("tp_gather", obj, root)
        return result if self.tp_rank == root else None

    def tp_allreduce(self, obj: object, op: ReduceOp = ReduceOp.SUM) -> object:
        operations = {
            ReduceOp.SUM: operator.add,
            ReduceOp.PRODUCT: operator.mul,
            ReduceOp.MIN: np.minimum,
            ReduceOp.MAX: np.maximum,
            ReduceOp.BAND: operator.and_,
            ReduceOp.BOR: operator.or_,
            ReduceOp.BXOR: operator.xor,
        }
        result = self._exchange("tp_allreduce", obj, op)
        return reduce(operations[op], result)

    def broadcast(self, obj: object, root: int = 0) -> object:
        if not 0 <= root < self.tp_size:
            raise ValueError("root outside TP group")
        return self._exchange("broadcast", obj, root)[root]

    def broadcast_int64(self, values: object, root: int = 0) -> np.ndarray:
        if not 0 <= root < self.tp_size:
            raise ValueError("root outside TP group")
        vector = np.asarray(values, dtype=np.int64).reshape(-1)
        result = self._exchange("broadcast_int64", vector, root, vector.size)
        return np.asarray(result[root], dtype=np.int64)
