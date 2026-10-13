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
"""Types that the DKV layer-split layout, the data plane and the KV cache manager share.

The module imports only the standard library, so it can be imported without a GPU.
"""

from typing import NamedTuple


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
