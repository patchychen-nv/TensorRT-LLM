# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-rank pool readers of the DKV integration tests follow each rank's own snapshots."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "integration/defs/dkv/dkv_stats.py"
_SPEC = importlib.util.spec_from_file_location("dkv_stats_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_STATS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STATS)

pytestmark = pytest.mark.cpu_only


def _snapshot(rank: int, *free_per_pool: list[int]) -> dict:
    """A rank's snapshot with one cache level per argument and one pool per free count."""
    return {
        "rank": rank,
        "pools_by_level": [[{"free": free} for free in level] for level in free_per_pool],
    }


def _row(iteration: int, snapshot: dict | None) -> dict:
    row: dict = {"iter": iteration}
    if snapshot is not None:
        row["dkvMeasurement"] = snapshot
    return row


def test_free_pages_are_summed_over_every_pool_of_every_level() -> None:
    assert _STATS.pool_free_pages(_snapshot(0, [3, 4], [10])) == 17


def test_each_rank_reports_its_own_newest_snapshot() -> None:
    rows = [
        _row(1, _snapshot(0, [8])),
        _row(1, _snapshot(1, [8])),
        _row(2, _snapshot(0, [7])),
        _row(2, _snapshot(1, [5])),
    ]
    assert _STATS.latest_free_pages_by_rank(rows) == {0: 7, 1: 5}


def test_the_newest_snapshot_wins_whatever_the_row_order() -> None:
    rows = [_row(5, _snapshot(0, [1])), _row(3, _snapshot(0, [9]))]
    assert _STATS.latest_free_pages_by_rank(rows) == {0: 1}


def test_rows_without_a_snapshot_are_ignored() -> None:
    rows = [_row(1, None), _row(2, _snapshot(1, [6])), _row(3, None)]
    assert _STATS.latest_free_pages_by_rank(rows) == {1: 6}
    assert _STATS.latest_free_pages_by_rank([]) == {}
