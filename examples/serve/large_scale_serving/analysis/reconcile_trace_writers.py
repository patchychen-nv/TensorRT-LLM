#!/usr/bin/env python3
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
"""Reconcile a request-trace directory against its writers' own accounting.

Every RequestTraceWriter publishes ``<trace dir>/_writer/<generation>.json``
(see tensorrt_llm/serve/request_trace.py). For each generation this checks:

- the books: ``submitted == persisted + sum(dropped) + unknown + pending``;
- every shard the generation claims: the bytes it says its durable appends
  occupy (``start_offset``..``persisted_end``), minus the extents of its failed
  appends, hold exactly ``persisted_records`` parseable lines and no bad line.

It also counts what lies in each shard outside every generation's durable
range: lines appended after the last published sidecar, or by a writer that
never published one. Nothing is inferred about those; they are reported as
unaccounted.

Runs on the cluster against the original, uncompressed shards; the offsets in a
sidecar are offsets into those files. Standard library only.

usage: reconcile_trace_writers.py <trace dir> [--json out.json]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

SIDECAR_DIR = "_writer"


def verify_shard(trace_dir, shard):
    """Same check as request_trace.verify_writer_shards, for one shard.

    Kept in step with it: test_request_trace_durability.py runs both on the
    same writer output and requires identical results.
    """
    path = os.path.join(trace_dir, shard["file"])
    result = dict(shard, records=0, bad_lines=0, unknown_lines=0, size=None, short=False)
    begin, end = shard["start_offset"], shard["persisted_end"]
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            result["size"] = handle.tell()
            handle.seek(begin)
            data = bytearray(handle.read(end - begin))
    except OSError as error:
        result["error"] = f"{type(error).__name__}: {error}"
        return result
    result["short"] = len(data) < end - begin
    for extent in shard.get("unknown_extents", []):
        lo = extent["start"] - begin
        hi = min(len(data), lo + extent["landed"])
        if lo < 0 or lo >= len(data):
            continue
        result["unknown_lines"] += bytes(data[lo:hi]).count(b"\n") + (
            1 if hi > lo and data[hi - 1 : hi] != b"\n" else 0
        )
        data[lo:hi] = b"\n" * (hi - lo)
    for line in bytes(data).split(b"\n"):
        if not line:
            continue
        try:
            json.loads(line)
            result["records"] += 1
        except ValueError:
            result["bad_lines"] += 1
    return result


def unclaimed_lines(data, claimed):
    """Lines in ``data`` outside every ``(begin, end)`` range in ``claimed``.

    Returns (whole lines, 1 if a line is cut off at a range boundary or at
    the end of the file else 0, bytes). Lone newlines -- the ones a writer
    writes to isolate a fragment -- are not lines.
    """
    segments, cursor = [], 0
    for begin, end in sorted((max(0, b), min(e, len(data))) for b, e in claimed):
        if begin > cursor:
            segments.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < len(data):
        segments.append((cursor, len(data)))
    whole = partial = size = 0
    for begin, end in segments:
        pieces = data[begin:end].split(b"\n")
        whole += sum(1 for piece in pieces[:-1] if piece)
        partial += 1 if pieces[-1] else 0
        size += end - begin
    return whole, partial, size


def reconcile(trace_dir):
    generations = []
    claimed = defaultdict(list)  # shard file -> [(begin, end)]
    for path in sorted(glob.glob(os.path.join(trace_dir, SIDECAR_DIR, "*.json"))):
        with open(path) as handle:
            sidecar = json.load(handle)
        counts = sidecar["counts"]
        books = (
            counts["persisted"]
            + sum(counts["dropped"].values())
            + counts["unknown"]
            + counts["pending"]
        )
        shards = [verify_shard(trace_dir, shard) for shard in sidecar.get("shards", [])]
        for shard in sidecar.get("shards", []):
            claimed[shard["file"]].append((shard["start_offset"], shard["persisted_end"]))
            # A failed append's bytes are the sidecar's to account for (as
            # unknown), not unaccounted, even past the durable range.
            for extent in shard.get("unknown_extents", []):
                claimed[shard["file"]].append((extent["start"], extent["start"] + extent["landed"]))
        generations.append(
            {
                "generation": sidecar["generation"],
                "sidecar": os.path.relpath(path, trace_dir),
                "published_at": sidecar["published_at"],
                "closed": sidecar["closed"],
                "counts": counts,
                "attributes": sidecar.get("attributes", {}),
                "errors": sidecar.get("errors", {}),
                "books_balance": counts["submitted"] == books and counts["pending"] >= 0,
                # The counts and the watermarks are booked together, so the
                # persisted records of all its shards add up to its count.
                "shards_match_counts": sum(s["persisted_records"] for s in shards)
                == counts["persisted"],
                "shards_verify": all(
                    s.get("error") is None
                    and not s["short"]
                    and s["records"] == s["persisted_records"]
                    and s["bad_lines"] == 0
                    for s in shards
                ),
                "shards": shards,
            }
        )
    # Bytes no generation claims as durable, per shard.
    unaccounted = {}
    for path in sorted(glob.glob(os.path.join(trace_dir, "*", "*.jsonl"))):
        relative = os.path.relpath(path, trace_dir)
        with open(path, "rb") as handle:
            data = handle.read()
        whole, partial, size = unclaimed_lines(data, claimed.get(relative, []))
        if whole or partial:
            unaccounted[relative] = {
                "size": len(data),
                "unclaimed_bytes": size,
                "whole_lines": whole,
                "cut_lines": partial,
            }
    return {
        "trace_dir": trace_dir,
        "generations": generations,
        "all_books_balance": all(
            g["books_balance"] and g["shards_match_counts"] for g in generations
        ),
        "all_shards_verify": all(g["shards_verify"] for g in generations),
        "unaccounted": unaccounted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_dir")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args()
    report = reconcile(args.trace_dir)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=1)
    summary = {
        "generations": len(report["generations"]),
        "all_books_balance": report["all_books_balance"],
        "all_shards_verify": report["all_shards_verify"],
        "unaccounted_shards": len(report["unaccounted"]),
    }
    print(json.dumps(summary))
    return 0 if report["all_books_balance"] and report["all_shards_verify"] else 1


if __name__ == "__main__":
    sys.exit(main())
