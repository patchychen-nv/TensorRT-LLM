# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each rank's own KV pool state, read from the iteration stats of an attention-DP run.

Every iteration row of an attention-DP run carries the ``kvCacheStats`` of rank 0, so a check built
on them cannot see another replica drift. With ``TRTLLM_DKV_MEASUREMENT=1`` every row also carries
the exporting rank's own pool snapshot as ``dkvMeasurement``.
"""


def pool_free_pages(snapshot: dict) -> int:
    """Free pages summed over every pool of every cache level in one rank's snapshot."""
    return sum(pool["free"] for level in snapshot["pools_by_level"] for pool in level)


def latest_free_pages_by_rank(rows: list[dict]) -> dict[int, int]:
    """The free pages each rank reported in its newest snapshot among ``rows``."""
    latest: dict[int, tuple[int, dict]] = {}
    for row in rows:
        snapshot = row.get("dkvMeasurement")
        if snapshot is not None and row["iter"] >= latest.get(snapshot["rank"], (-1, None))[0]:
            latest[snapshot["rank"]] = (row["iter"], snapshot)
    return {rank: pool_free_pages(snapshot) for rank, (_, snapshot) in sorted(latest.items())}
