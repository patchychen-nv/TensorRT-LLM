# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Corrupt the KV bytes of one request on their way out of a DKV context worker.

A precision gate is only worth something if it fails when the KV a request hands to generation is
wrong. This hook makes that happen on purpose, in the context worker's own process and without
editing the transfer code: it wraps ``Sender._deliver_kv_to_agent`` and, for the request whose prompt
has the configured length, overwrites the source regions of every write before the transport reads
them. The spec is a JSON object:

``kind``       ``"zero"`` fills the regions with zero bytes; ``"foreign"`` fills each region with a
               region of the same size from the previous request that went to the same generation
               rank, which is another request's KV in the same layout.
``prompt_len`` token count of the one prompt to corrupt; it must be unique among the submitted
               prompts.
``fraction``   share of the transferred regions to corrupt, taken from the front of the transfer
               order (default 1).
``owners``     ranks of the context group whose senders corrupt what they send (default: all). In a
               layer-split group every rank sends the layers it owns, so one owner can be corrupted
               alone.
``record_dir`` directory that receives one JSON line per corrupted write.
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
from collections import defaultdict
from functools import wraps
from pathlib import Path

_TARGET_MODULE = "tensorrt_llm._torch.disaggregation.native.transfer"
_KINDS = ("zero", "foreign")


def _copy_device(module, destination: int, source: int, size: int) -> None:
    cudart = module.cudart
    module.CUASSERT(
        cudart.cudaMemcpy(destination, source, size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    )


def _save_payload(module, pointers: list[int], sizes: list[int]):
    """Copy the regions of a write into one device buffer.

    Returns the buffer and, for every region size, the offsets in the buffer of the regions of that
    size.
    """
    import torch

    saved = torch.empty(sum(sizes), dtype=torch.uint8, device="cuda")
    offsets_by_size: dict[int, list[int]] = defaultdict(list)
    offset = 0
    for pointer, size in zip(pointers, sizes):
        _copy_device(module, saved.data_ptr() + offset, pointer, size)
        offsets_by_size[size].append(offset)
        offset += size
    module.CUASSERT(module.cudart.cudaDeviceSynchronize())
    return saved, offsets_by_size


def _zero(module, pointers: list[int], sizes: list[int]) -> int:
    for pointer, size in zip(pointers, sizes):
        module.CUASSERT(module.cudart.cudaMemset(pointer, 0, size))
    return sum(sizes)


def _overwrite_with_foreign(module, pointers: list[int], sizes: list[int], payload) -> int:
    """Fill each region with a region of the same size of the other request's write.

    The n-th region of a size takes the n-th region of that size of the other write, wrapping around
    when this write has more of them. Regions without a counterpart keep their bytes.
    """
    saved, offsets_by_size = payload
    used: dict[int, int] = defaultdict(int)
    replaced = 0
    for pointer, size in zip(pointers, sizes):
        offsets = offsets_by_size.get(size)
        if not offsets:
            continue
        offset = offsets[used[size] % len(offsets)]
        used[size] += 1
        _copy_device(module, pointer, saved.data_ptr() + offset, size)
        replaced += size
    return replaced


def install_sender_mutation(module, spec: dict) -> None:
    """Wrap the sender of an imported transfer module according to ``spec``."""
    if spec["kind"] not in _KINDS:
        raise ValueError(f"Unknown KV mutation kind {spec['kind']!r}; expected one of {_KINDS}")
    fraction = float(spec.get("fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("The corrupted fraction of the KV regions must be in (0, 1]")
    owners = spec.get("owners")
    if owners is not None and not all(isinstance(owner, int) for owner in owners):
        raise ValueError("The owners of a KV mutation are the integer ranks of the context group")
    sender = module.Sender
    if getattr(sender, "_dkv_mutation_installed", False):
        return
    sender._dkv_mutation_installed = True
    record_dir = Path(spec["record_dir"])
    record_dir.mkdir(parents=True, exist_ok=True)
    # The regions of the last other request per generation rank, kept for the "foreign" kind.
    previous: dict[int, tuple] = {}
    lock = threading.Lock()
    original = sender._deliver_kv_to_agent

    def mutate(write_meta, owner: int) -> None:
        pointers = write_meta.src_ptrs.tolist()
        sizes = write_meta.sizes.tolist()
        # The forward that wrote this KV may still be running on another stream.
        module.CUASSERT(module.cudart.cudaDeviceSynchronize())
        if getattr(write_meta.task, "_prompt_len", None) != spec["prompt_len"]:
            if spec["kind"] == "foreign":
                with lock:
                    previous[write_meta.peer_rank] = _save_payload(module, pointers, sizes)
            return
        count = max(1, round(len(pointers) * fraction))
        if spec["kind"] == "zero":
            changed = _zero(module, pointers[:count], sizes[:count])
        else:
            with lock:
                payload = previous.get(write_meta.peer_rank)
            if payload is None:
                raise RuntimeError("No earlier request went to this generation rank to borrow from")
            changed = _overwrite_with_foreign(module, pointers[:count], sizes[:count], payload)
        module.CUASSERT(module.cudart.cudaDeviceSynchronize())
        row = {
            "kind": spec["kind"],
            "prompt_len": spec["prompt_len"],
            "unique_rid": write_meta.unique_rid,
            "owner": owner,
            "peer_rank": write_meta.peer_rank,
            "regions": len(pointers),
            "corrupted_regions": count,
            "corrupted_bytes": changed,
        }
        with (record_dir / f"mutations-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    @wraps(original)
    def deliver(self, write_meta):
        if write_meta.meta_type == module.WriteMetaType.KV and write_meta.src_ptrs.size > 0:
            if owners is None or self._instance_rank in owners:
                mutate(write_meta, self._instance_rank)
        return original(self, write_meta)

    sender._deliver_kv_to_agent = deliver


class _MutationLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, spec: dict) -> None:
        self.wrapped = wrapped
        self.spec = spec

    def create_module(self, spec):
        return self.wrapped.create_module(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        install_sender_mutation(module, self.spec)


class _MutationFinder(importlib.abc.MetaPathFinder):
    def __init__(self, spec: dict) -> None:
        self.spec = spec

    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET_MODULE:
            return None
        found = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if found is not None:
            found.loader = _MutationLoader(found.loader, self.spec)
        return found


def install_mutation_hook(spec_json: str) -> None:
    """Install before TensorRT-LLM imports in each context MPI worker."""
    sys.meta_path.insert(0, _MutationFinder(json.loads(spec_json)))
