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

#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void DiskCacheTierConfig::assertValid() const
{
    if (quota == 0)
    {
        throw std::invalid_argument("DiskCacheTierConfig: quota must be > 0");
    }
    if (!std::filesystem::is_directory(path))
    {
        throw std::invalid_argument("DiskCacheTierConfig: path '" + path + "' is not a directory");
    }
}

void KVCacheManagerConfig::validate() const
{
    if (swaScratchReuse.has_value())
    {
        swaScratchReuse->validate();
    }
    for (auto const& batch : constraints)
    {
        batch.validate();
    }
    if (typicalStep.has_value())
    {
        typicalStep->validate();
    }

    // These mirror Python's KVCacheManagerConfig.__post_init__ asserts, so they
    // throw AssertionError (translated in the binding layer) rather than ValueError.
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw AssertionError("KVCacheManagerConfig: first cache tier must be GPU memory");
    }

    // The life cycle registry does not exist yet, so only the shape and the values can be checked
    // here. The number of life cycles is checked where the registry is available, in the
    // StorageManager constructor.
    if (lifecycleSlotCounts.has_value())
    {
        if (initialPoolRatio.has_value())
        {
            throw std::invalid_argument("lifecycle_slot_counts and initial_pool_ratio are mutually exclusive");
        }
        if (lifecycleSlotCounts->size() != cacheTiers.size())
        {
            throw std::invalid_argument("lifecycle_slot_counts must have one row per cache tier");
        }
        size_t const numLifeCycles = lifecycleSlotCounts->front().size();
        for (auto const& row : *lifecycleSlotCounts)
        {
            if (row.empty() || row.size() != numLifeCycles)
            {
                throw std::invalid_argument("lifecycle_slot_counts rows must be non-empty and have the same length");
            }
            // Page indices are 32-bit throughout the storage layer, so a larger count cannot be addressed.
            if (std::any_of(row.begin(), row.end(),
                    [](std::int64_t count) { return count <= 0 || count > std::numeric_limits<int>::max(); }))
            {
                throw std::invalid_argument(
                    "lifecycle_slot_counts values must be positive and fit a 32-bit page index");
            }
        }
    }

    // Check for duplicate layer ids.
    std::set<LayerId> seenLayerIds;
    for (auto const& layer : layers)
    {
        std::visit(
            [&](auto const& cfg)
            {
                cfg.validate();
                if (!seenLayerIds.insert(cfg.layerId).second)
                {
                    throw AssertionError("KVCacheManagerConfig: duplicate layer id");
                }
                for (auto const& buf : cfg.buffers)
                {
                    if (buf.isSparse && (cacheTiers.size() < 2 || cacheTierOf(cacheTiers[1]) != CacheTier::HOST_MEM))
                    {
                        throw std::invalid_argument("Sparse buffers require cache level 1 to use HOST_MEM");
                    }
                    if (buf.tokensPerBlockOverride.has_value()
                        && (*buf.tokensPerBlockOverride <= 0 || tokensPerBlock % *buf.tokensPerBlockOverride != 0))
                    {
                        throw AssertionError(
                            "KVCacheManagerConfig: tokensPerBlockOverride must be a divisor of "
                            "tokensPerBlock");
                    }
                }
            },
            layer);
    }

    // SSM-specific validation.
    bool hasSSM = false;
    for (auto const& layer : layers)
    {
        if (std::holds_alternative<SsmLayerConfig>(layer))
        {
            hasSSM = true;
            break;
        }
    }
    if (hasSSM)
    {
        if (!commitMinSnapshot)
            throw AssertionError("KVCacheManagerConfig: commit_min_snapshot must be True when SSM layers are present");
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
