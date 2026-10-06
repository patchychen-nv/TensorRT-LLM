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

#include "kvCacheManagerV2TestUtils.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeConfig;
using tensorrt_llm::batch_manager::kv_cache_manager_v2::test::makeTieredConfig;

using SlotCounts = std::vector<std::vector<std::int64_t>>;

AttentionLayerConfig makeAttentionLayer(LayerId layerId, bool isSparse, size_t bufferSize = 4096)
{
    return AttentionLayerConfig{
        .layerId = layerId, .buffers = {BufferConfig{.role = "key", .size = bufferSize, .isSparse = isSparse}}};
}

TEST(KvCacheManagerV2ConfigTest, ExistingBufferConfigurationDefaultsToDense)
{
    BufferConfig const buffer{"key", 4096, 2};
    EXPECT_FALSE(buffer.isSparse);
    EXPECT_EQ(buffer.tokensPerBlockOverride, 2);
    EXPECT_NO_THROW(makeConfig().validate());
}

TEST(KvCacheManagerV2ConfigTest, SparseBuffersRequireHostMemoryAtLevelOne)
{
    auto config = makeConfig();
    config.layers = {makeAttentionLayer(0, true)};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers.emplace_back(DiskCacheTierConfig{4 << 20, "/tmp"});
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers[1] = GpuCacheTierConfig{4 << 20};
    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers[1] = HostCacheTierConfig{4 << 20};
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, MixedBuffersInOneLayerAreRejected)
{
    auto config = makeTieredConfig();
    for (bool const firstIsSparse : {false, true})
    {
        auto layer = makeAttentionLayer(0, firstIsSparse);
        layer.buffers.push_back(BufferConfig{.role = "value", .size = 4096, .isSparse = !firstIsSparse});
        config.layers = {layer};
        EXPECT_THROW(config.validate(), std::invalid_argument);
        EXPECT_THROW(createStorageConfig(config), std::invalid_argument);
    }
}

TEST(KvCacheManagerV2ConfigTest, SparseSsmBuffersAreRejected)
{
    auto config = makeTieredConfig();
    config.commitMinSnapshot = true;
    config.layers
        = {SsmLayerConfig{.layerId = 0, .buffers = {BufferConfig{.role = "state", .size = 4096, .isSparse = true}}}};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    EXPECT_THROW(createStorageConfig(config), std::invalid_argument);
}

TEST(KvCacheManagerV2ConfigTest, SparseAndDenseBuffersHaveSeparateLifecyclesAndPools)
{
    auto config = makeTieredConfig();
    config.layers = {makeAttentionLayer(0, false), makeAttentionLayer(1, true), makeAttentionLayer(2, false),
        makeAttentionLayer(3, true)};
    ASSERT_NO_THROW(config.validate());

    LifeCycleRegistry const registry(config);
    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto const& dense = attributes.at(BufferId{0, "key"});
    auto const& sparse = attributes.at(BufferId{1, "key"});

    EXPECT_EQ(registry.size(), LifeCycleId{2});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_NE(dense.lifeCycleId, sparse.lifeCycleId);
    EXPECT_NE(grouping[dense.lifeCycleId], grouping[sparse.lifeCycleId]);
    EXPECT_FALSE(std::get<AttnLifeCycle>(registry[dense.lifeCycleId]).isSparse);
    EXPECT_TRUE(std::get<AttnLifeCycle>(registry[sparse.lifeCycleId]).isSparse);
    EXPECT_FALSE(registry[dense.lifeCycleId] == registry[sparse.lifeCycleId]);

    auto const& otherDense = attributes.at(BufferId{2, "key"});
    auto const& otherSparse = attributes.at(BufferId{3, "key"});
    EXPECT_EQ(dense.lifeCycleId, otherDense.lifeCycleId);
    EXPECT_EQ(sparse.lifeCycleId, otherSparse.lifeCycleId);
    EXPECT_EQ(sparse.poolIndex, otherSparse.poolIndex);
    EXPECT_EQ(otherSparse.offset, sparse.offset + sparse.size);
    EXPECT_EQ(otherDense.offset, dense.offset + dense.size);
}

TEST(KvCacheManagerV2ConfigTest, MatchingSlotShapesSharePoolsWithinEachSparsity)
{
    auto config = makeTieredConfig();
    config.layers.clear();
    for (LayerId layerId = 0; layerId < 4; ++layerId)
    {
        auto layer = makeAttentionLayer(layerId, layerId >= 2);
        layer.slidingWindowSize = layerId % 2 == 0 ? 128 : 256;
        config.layers.push_back(layer);
    }
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const grouping = storage.lifeCycleGrouping();
    auto poolGroup = [&](LayerId layerId) { return grouping[attributes.at(BufferId{layerId, "key"}).lifeCycleId]; };
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{4});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_EQ(poolGroup(0), poolGroup(1));
    EXPECT_EQ(poolGroup(2), poolGroup(3));
    EXPECT_NE(poolGroup(0), poolGroup(2));
}

//! Three dense attention life cycles (windows 128 and 256, and no window) whose slots all have the same
//! size, over a GPU and a host tier. Without fixed page counts they share one pool group.
KVCacheManagerConfig makeEqualSlotLifeCycleConfig()
{
    auto config = makeTieredConfig();
    config.layers.clear();
    for (LayerId layerId = 0; layerId < 3; ++layerId)
    {
        auto layer = makeAttentionLayer(layerId, false);
        if (layerId < 2)
        {
            layer.slidingWindowSize = layerId == 0 ? 128 : 256;
        }
        config.layers.push_back(layer);
    }
    return config;
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsAreOffByDefault)
{
    auto const config = makeEqualSlotLifeCycleConfig();
    EXPECT_FALSE(config.lifecycleSlotCounts.has_value());
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsMustFormOneRectangularPositiveRowPerCacheTier)
{
    auto config = makeEqualSlotLifeCycleConfig();
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, 60}};
    EXPECT_NO_THROW(config.validate());

    // One row per cache tier.
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.lifecycleSlotCounts = SlotCounts{};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    // Rows are non-empty and equally long.
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50}};
    EXPECT_THROW(config.validate(), std::invalid_argument);
    config.lifecycleSlotCounts = SlotCounts{{}, {}};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    // Counts are positive and addressable by a 32-bit page index.
    for (std::int64_t const invalid : {std::int64_t{0}, std::int64_t{-1}, std::int64_t{1} << 31})
    {
        SCOPED_TRACE(invalid);
        config.lifecycleSlotCounts = SlotCounts{{10, invalid, 30}, {40, 50, 60}};
        EXPECT_THROW(config.validate(), std::invalid_argument);
        config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, invalid}};
        EXPECT_THROW(config.validate(), std::invalid_argument);
    }
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, std::numeric_limits<int>::max()}};
    EXPECT_NO_THROW(config.validate());

    // The number of life cycles is only known once the registry exists, so a row of the wrong length is
    // rejected by the StorageManager and not here.
    config.lifecycleSlotCounts = SlotCounts{{10, 20}, {40, 50}};
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsExcludeInitialPoolRatio)
{
    auto config = makeEqualSlotLifeCycleConfig();
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, 60}};
    config.initialPoolRatio = std::vector<float>{0.2F, 0.3F, 0.5F};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.lifecycleSlotCounts.reset();
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsKeepTheSparseHostTierRequirement)
{
    auto config = makeEqualSlotLifeCycleConfig();
    config.layers = {makeAttentionLayer(0, true)};
    config.cacheTiers = {GpuCacheTierConfig{4 << 20}};
    config.lifecycleSlotCounts = SlotCounts{std::vector<std::int64_t>{10}};
    EXPECT_THROW(config.validate(), std::invalid_argument);

    config.cacheTiers.emplace_back(HostCacheTierConfig{4 << 20});
    config.lifecycleSlotCounts = SlotCounts{{10}, {10}};
    EXPECT_NO_THROW(config.validate());
}

TEST(KvCacheManagerV2ConfigTest, EqualSlotLifeCyclesShareOnePoolGroupWithoutLifecycleSlotCounts)
{
    auto const config = makeEqualSlotLifeCycleConfig();
    auto const storage = createStorageConfig(config);
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{3});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
    EXPECT_EQ(storage.slotDescList[PoolGroupIndex{0}].variants.size(), 3U);
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsGiveEveryLifeCycleItsOwnPoolGroup)
{
    auto config = makeEqualSlotLifeCycleConfig();
    config.lifecycleSlotCounts = SlotCounts{{10, 20, 30}, {40, 50, 60}};

    auto const storage = createStorageConfig(config);
    auto const grouping = storage.lifeCycleGrouping();
    ASSERT_EQ(storage.numLifeCycles(), LifeCycleId{3});
    ASSERT_EQ(storage.slotDescList.size(), PoolGroupIndex{3});
    for (LifeCycleId lifeCycle{0}; lifeCycle < LifeCycleId{3}; ++lifeCycle)
    {
        PoolGroupIndex const poolGroup = grouping[lifeCycle];
        EXPECT_EQ(poolGroup.value(), lifeCycle.value());
        ASSERT_EQ(storage.slotDescList[poolGroup].variants.size(), 1U);
        EXPECT_EQ(storage.slotDescList[poolGroup].variants.front().lifeCycleId, lifeCycle);
    }

    // The buffer layout of a life cycle does not change, only which pool group it lives in.
    auto const attributes = storage.bufferAttributes();
    for (LayerId layerId = 0; layerId < 3; ++layerId)
    {
        auto const& attribute = attributes.at(BufferId{layerId, "key"});
        EXPECT_EQ(attribute.lifeCycleId.value(), layerId);
        EXPECT_EQ(attribute.poolIndex, PoolIndex{0});
        EXPECT_EQ(attribute.offset, 0U);
    }
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsOrderPoolGroupsByLifeCycleNotBySlotSize)
{
    // Life cycle 0 (window 128) has the larger slot and life cycle 1 (no window) the smaller one. The default
    // grouping sorts pool groups by slot size, which puts them in the opposite order.
    auto config = makeTieredConfig();
    auto first = makeAttentionLayer(0, false, 8192);
    first.slidingWindowSize = 128;
    config.layers = {first, makeAttentionLayer(1, false, 4096)};

    auto const byteOrder = createStorageConfig(config);
    EXPECT_EQ(byteOrder.lifeCycleGrouping()[LifeCycleId{0}], PoolGroupIndex{1});
    EXPECT_EQ(byteOrder.lifeCycleGrouping()[LifeCycleId{1}], PoolGroupIndex{0});

    config.lifecycleSlotCounts = SlotCounts{{10, 20}, {40, 50}};
    auto const lifeCycleOrder = createStorageConfig(config);
    EXPECT_EQ(lifeCycleOrder.lifeCycleGrouping()[LifeCycleId{0}], PoolGroupIndex{0});
    EXPECT_EQ(lifeCycleOrder.lifeCycleGrouping()[LifeCycleId{1}], PoolGroupIndex{1});
    EXPECT_EQ(lifeCycleOrder.slotDescList[PoolGroupIndex{0}].slotSizeList()[PoolIndex{0}], 8192U);
    EXPECT_EQ(lifeCycleOrder.slotDescList[PoolGroupIndex{1}].slotSizeList()[PoolIndex{0}], 4096U);
}

TEST(KvCacheManagerV2ConfigTest, LifecycleSlotCountsSeparateSparseAndDenseLifeCyclesOfEqualSlotSize)
{
    auto config = makeTieredConfig();
    config.layers = {makeAttentionLayer(0, false), makeAttentionLayer(1, true)};
    config.lifecycleSlotCounts = SlotCounts{{10, 20}, {40, 50}};
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{2});
    EXPECT_NE(storage.lifeCycleGrouping()[LifeCycleId{0}], storage.lifeCycleGrouping()[LifeCycleId{1}]);
}

TEST(KvCacheManagerV2ConfigTest, SparseBuffersCoalesceByExpandedSize)
{
    auto config = makeTieredConfig();
    auto firstLayer = makeAttentionLayer(0, true, 2048);
    firstLayer.buffers.front().tokensPerBlockOverride = 2;
    config.layers = {firstLayer, makeAttentionLayer(1, true, 4096)};
    ASSERT_NO_THROW(config.validate());

    auto const storage = createStorageConfig(config);
    auto const attributes = storage.bufferAttributes();
    auto const& first = attributes.at(BufferId{0, "key"});
    auto const& second = attributes.at(BufferId{1, "key"});
    EXPECT_EQ(storage.numLifeCycles(), LifeCycleId{1});
    EXPECT_EQ(storage.slotDescList.size(), PoolGroupIndex{1});
    EXPECT_EQ(first.lifeCycleId, second.lifeCycleId);
    EXPECT_EQ(first.poolIndex, second.poolIndex);
    EXPECT_EQ(first.size, second.size);
    EXPECT_EQ(second.offset, first.offset + first.size);
    EXPECT_EQ(first.expansion, 2);
    EXPECT_EQ(second.expansion, 1);
}

} // namespace
