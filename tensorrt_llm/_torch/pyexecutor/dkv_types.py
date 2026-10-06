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
"""Types that the DKV layer-split layout, plan and data plane share.

The module imports only the standard library, so the plan (``dkv_plan``) can be built and checked
without a GPU, and the staging layout (``dkv_staging``) and the KV cache manager name the same things.
"""

import enum
from typing import NamedTuple


class StagingKind(enum.IntEnum):
    """One kind of KV storage of a layer, in the order the segments of a message list them.

    ``SWA`` is the sliding-window KV, ``COMPRESS_R4`` and ``COMPRESS_R128`` the compressed KV of
    the layers with compression ratio 4 and 128, ``INDEXER_COMPRESS`` the indexer's compressed
    keys, ``STATE_CSA``, ``STATE_HCA`` and ``STATE_INDEXER`` the read-modify-write states of the
    compressor of the two layer types and of the indexer. A state with several cache roles counts as
    one kind.
    """

    SWA = 0
    COMPRESS_R4 = 1
    COMPRESS_R128 = 2
    INDEXER_COMPRESS = 3
    STATE_CSA = 4
    STATE_HCA = 5
    STATE_INDEXER = 6

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def shared_page_index(self) -> bool:
        """Whether every layer indexes pages with the same page indices.

        The compressed caches are indexed alike by all layers that have them, and a layer's own
        region of the slot is selected by the base pointer. The sliding-window caches fold the
        layer into the page index instead and share one base pointer.
        """
        return self in (
            StagingKind.COMPRESS_R4,
            StagingKind.COMPRESS_R128,
            StagingKind.INDEXER_COMPRESS,
        )

    @property
    def windowed(self) -> bool:
        """Whether only the last tokens of the history are kept (a sliding window)."""
        return not self.shared_page_index

    @property
    def compress_ratio(self) -> int | None:
        """The compress ratio of the layers that have this kind; ``None``: every layer."""
        return {
            StagingKind.SWA: None,
            StagingKind.COMPRESS_R4: 4,
            StagingKind.COMPRESS_R128: 128,
            StagingKind.INDEXER_COMPRESS: 4,
            StagingKind.STATE_CSA: 4,
            StagingKind.STATE_HCA: 128,
            StagingKind.STATE_INDEXER: 4,
        }[self]


class LifecycleKey(NamedTuple):
    """Semantic identity of a KV life cycle, the same on every rank whatever layers it holds.

    Life cycle ids are assigned in the order the layers of a rank first mention each life cycle,
    so they differ between ranks that hold different layer subsets; this key does not.
    ``window_size`` is 0 for a life cycle that keeps the whole history.
    """

    is_ssm: bool
    window_size: int
    num_sink_blocks: int
    is_sparse: bool
