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
"""Transport of the DKV ``layer_split`` data plane: byte messages between two ranks of the group.

A transport moves a buffer from one rank to another on a CUDA stream. ``send`` and ``recv`` only
enqueue the operation, so they return before the data has moved, and the operation runs when the
stream gets to it. The two ends of a message are matched by the order of the calls between the two
ranks, not by a tag: the n-th ``send`` to a peer pairs with the n-th ``recv`` from that rank, and
both buffers must have the same size. Whether a send may finish before its receive is posted is up
to the implementation, so a caller must not rely on it (see ``dkv_plan``, which orders the messages
so that no rank waits on a message that its peer has not yet reached).
"""

from collections.abc import Sequence
from typing import Protocol

import torch


class DkvTransport(Protocol):
    """What the data plane needs from the interconnect."""

    def send(self, buffer: torch.Tensor, peer: int, stream: torch.cuda.Stream) -> None:
        """Enqueue the transfer of the bytes of ``buffer`` to ``peer`` on ``stream``.

        The buffer must stay untouched until the work of ``stream`` that follows has run.
        """
        ...

    def recv(self, buffer: torch.Tensor, peer: int, stream: torch.cuda.Stream) -> None:
        """Enqueue the reception of ``buffer.numel()`` bytes from ``peer`` on ``stream``."""
        ...


Message = tuple[torch.Tensor, int]
"""A buffer and the peer it goes to or comes from."""


class NcclP2PTransport:
    """Point-to-point messages over a communicator of its own.

    The communicator is created with the world size and the rank, so every rank of the world has to
    construct the transport at the same time, from the thread that runs the collective setup of the
    process and before any other communicator is active on the device. Creating it blocks until all
    ranks have arrived. The ranks of the DKV group are the ranks of the world.

    ``group_send_recv`` issues all the messages of a step in one NCCL group, which NCCL runs as one
    kernel. It is ``None`` on a native build whose communicator op has no group call; the streamer
    then sends and receives one message at a time.
    """

    def __init__(self, world_size: int, rank: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in a world of {world_size} ranks")
        self.world_size = world_size
        self.rank = rank
        self._comm = torch.classes.trtllm.NcclCommunicatorOp(world_size, rank)
        if not hasattr(self._comm, "group_send_recv"):
            self.group_send_recv = None

    def send(self, buffer: torch.Tensor, peer: int, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            self._comm.send(buffer, peer)

    def recv(self, buffer: torch.Tensor, peer: int, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            self._comm.recv(buffer, peer)

    def group_send_recv(
        self, sends: Sequence[Message], recvs: Sequence[Message], stream: torch.cuda.Stream
    ) -> None:
        """Enqueue every send and receive of ``sends`` and ``recvs`` on ``stream`` as one group.

        The messages to one peer keep their order, so a peer that issues its side of them one at a
        time pairs them up the same way.
        """
        self._comm.group_send_recv(
            [buffer for buffer, _ in sends],
            [peer for _, peer in sends],
            [buffer for buffer, _ in recvs],
            [peer for _, peer in recvs],
            stream.cuda_stream,
        )

    def warmup(self, stream: torch.cuda.Stream) -> None:
        """Exchange one message between every two ranks so that no connection is made lazily.

        The pairs are visited in one global order, which every rank follows for the pairs it is in:
        a rank waits for its peer only in the pair that both reach first, so the exchange finishes
        even if a send does not complete before its receive is posted.
        """
        token = torch.zeros(16, dtype=torch.uint8, device="cuda")
        for low in range(self.world_size):
            for high in range(low + 1, self.world_size):
                if self.rank == low:
                    self.send(token, high, stream)
                    self.recv(token, high, stream)
                elif self.rank == high:
                    self.recv(token, low, stream)
                    self.send(token, low, stream)
        stream.synchronize()


_nccl_transports: dict[tuple[int, int], NcclP2PTransport] = {}


def nccl_p2p_transport(world_size: int, rank: int) -> NcclP2PTransport:
    """The transport of this process for a world, created at the first call.

    Destroying a communicator is collective: a rank that does it at a moment the others have not
    reached can deadlock them, which is what happens when several executors that share worker
    processes each build and drop one. The communicator therefore lives as long as the process and
    every executor of the world shares it. The first call is collective as well.
    """
    key = (world_size, rank)
    if key not in _nccl_transports:
        _nccl_transports[key] = NcclP2PTransport(world_size, rank)
    return _nccl_transports[key]
