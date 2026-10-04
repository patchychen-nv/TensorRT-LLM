# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV mutation hook corrupts exactly the request, and the regions, that its spec names."""

import ctypes
import importlib.util
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[3] / "integration/defs/dkv/dkv_mutation_probe.py"
_SPEC = importlib.util.spec_from_file_location("dkv_mutation_probe_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

pytestmark = pytest.mark.cpu_only


class _Cudart:
    """The CUDA runtime calls of the hook, acting on host memory."""

    class cudaMemcpyKind:
        cudaMemcpyDeviceToDevice = 3

    @staticmethod
    def cudaMemset(pointer, value, size):
        ctypes.memset(pointer, value, size)
        return (0,)

    @staticmethod
    def cudaMemcpy(destination, source, size, kind):
        ctypes.memmove(destination, source, size)
        return (0,)

    @staticmethod
    def cudaDeviceSynchronize():
        return (0,)


class _WriteMetaType(Enum):
    KV = "KV"
    AUX = "AUX"


@dataclass
class _Task:
    _prompt_len: int


@dataclass
class _WriteMeta:
    task: _Task
    unique_rid: int
    peer_rank: int
    src_ptrs: np.ndarray
    sizes: np.ndarray
    meta_type: _WriteMetaType = _WriteMetaType.KV


class _Sender:
    """Stands in for the transfer sender: remembers the bytes of every write it is handed."""

    delivered: list[list[bytes]]

    def __init__(self) -> None:
        self.delivered = []

    def _deliver_kv_to_agent(self, write_meta):
        self.delivered.append(_regions(write_meta))


def _regions(write_meta) -> list[bytes]:
    return [
        ctypes.string_at(int(pointer), int(size))
        for pointer, size in zip(write_meta.src_ptrs, write_meta.sizes)
    ]


def _module() -> SimpleNamespace:
    def cuassert(result):
        assert result[0] == 0
        return result[1:] or None

    return SimpleNamespace(
        Sender=type("Sender", (_Sender,), {}),
        WriteMetaType=_WriteMetaType,
        CUASSERT=cuassert,
        cudart=_Cudart,
    )


def _write(prompt_len: int, contents: list[bytes], *, peer: int = 0, rid: int = 1) -> _WriteMeta:
    """A write whose regions hold ``contents``; the buffers live as long as the write does."""
    buffers = [(ctypes.c_ubyte * len(content)).from_buffer_copy(content) for content in contents]
    write = _WriteMeta(
        _Task(prompt_len),
        rid,
        peer,
        np.array([ctypes.addressof(buffer) for buffer in buffers], dtype=np.int64),
        np.array([len(content) for content in contents], dtype=np.int64),
    )
    write.buffers = buffers
    return write


def _install(tmp_path: Path, **spec) -> tuple[SimpleNamespace, _Sender]:
    module = _module()
    _PROBE.install_sender_mutation(
        module, {"prompt_len": 130, "record_dir": str(tmp_path / "records"), **spec}
    )
    return module, module.Sender()


def _records(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((tmp_path / "records").glob("mutations-*.jsonl"))
        for line in path.read_text().splitlines()
    ]


@pytest.fixture(autouse=True)
def host_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook stages another request's bytes in a device tensor; here that tensor is on the host."""
    real_empty = torch.empty
    monkeypatch.setattr(
        torch, "empty", lambda size, dtype=None, device=None: real_empty(size, dtype=dtype)
    )


def test_zero_corrupts_only_the_prompt_of_the_configured_length(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="zero")
    other = _write(40, [b"\x07" * 8, b"\x08" * 8], rid=1)
    target = _write(130, [b"\x09" * 8, b"\x0a" * 4], rid=2)
    sender._deliver_kv_to_agent(other)
    sender._deliver_kv_to_agent(target)
    assert sender.delivered == [[b"\x07" * 8, b"\x08" * 8], [b"\x00" * 8, b"\x00" * 4]]
    (record,) = _records(tmp_path)
    assert record["unique_rid"] == 2 and record["peer_rank"] == 0
    assert record["regions"] == record["corrupted_regions"] == 2
    assert record["corrupted_bytes"] == 12


def test_fraction_corrupts_a_leading_share_of_the_regions(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="zero", fraction=0.3)
    sender._deliver_kv_to_agent(_write(130, [bytes([index + 1]) * 4 for index in range(10)]))
    (delivered,) = sender.delivered
    assert delivered[:3] == [b"\x00" * 4] * 3
    assert delivered[3:] == [bytes([index + 1]) * 4 for index in range(3, 10)]
    assert _records(tmp_path)[0]["corrupted_regions"] == 3


def test_a_tiny_fraction_still_corrupts_one_region(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="zero", fraction=0.001)
    sender._deliver_kv_to_agent(_write(130, [b"\x05" * 4, b"\x06" * 4]))
    assert sender.delivered == [[b"\x00" * 4, b"\x06" * 4]]


def test_every_write_of_the_request_is_corrupted_and_recorded(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="zero")
    for peer in (0, 1):
        sender._deliver_kv_to_agent(_write(130, [b"\x05" * 4], peer=peer))
    assert sender.delivered == [[b"\x00" * 4]] * 2
    assert sorted(record["peer_rank"] for record in _records(tmp_path)) == [0, 1]


def test_foreign_fills_regions_with_the_regions_of_the_same_size_of_the_previous_request(
    tmp_path: Path,
) -> None:
    _, sender = _install(tmp_path, kind="foreign")
    previous = [b"\x01" * 4, b"\x02" * 8, b"\x03" * 4]
    sender._deliver_kv_to_agent(_write(72, previous, rid=1))
    # The n-th region of a size takes the n-th of that size, wrapping around; a size the other
    # request never had keeps its own bytes.
    mine = [b"\x10" * 4, b"\x11" * 4, b"\x12" * 4, b"\x13" * 8, b"\x14" * 16]
    sender._deliver_kv_to_agent(_write(130, mine, rid=2))
    assert sender.delivered[1] == [
        b"\x01" * 4,
        b"\x03" * 4,
        b"\x01" * 4,
        b"\x02" * 8,
        b"\x14" * 16,
    ]
    (record,) = _records(tmp_path)
    assert record["corrupted_bytes"] == 4 + 4 + 4 + 8


def test_foreign_borrows_from_the_previous_request_of_the_same_generation_rank(
    tmp_path: Path,
) -> None:
    _, sender = _install(tmp_path, kind="foreign")
    sender._deliver_kv_to_agent(_write(72, [b"\x01" * 4], peer=0, rid=1))
    sender._deliver_kv_to_agent(_write(96, [b"\x02" * 4], peer=1, rid=2))
    sender._deliver_kv_to_agent(_write(130, [b"\x09" * 4], peer=1, rid=3))
    sender._deliver_kv_to_agent(_write(130, [b"\x09" * 4], peer=0, rid=3))
    assert sender.delivered[2:] == [[b"\x02" * 4], [b"\x01" * 4]]


def test_foreign_without_an_earlier_request_fails_loudly(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="foreign")
    with pytest.raises(RuntimeError, match="No earlier request"):
        sender._deliver_kv_to_agent(_write(130, [b"\x09" * 4]))


def test_writes_without_kv_regions_are_left_alone(tmp_path: Path) -> None:
    _, sender = _install(tmp_path, kind="zero")
    aux = _write(130, [b"\x09" * 4])
    aux.meta_type = _WriteMetaType.AUX
    empty = _WriteMeta(_Task(130), 3, 0, np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    sender._deliver_kv_to_agent(aux)
    sender._deliver_kv_to_agent(empty)
    assert sender.delivered == [[b"\x09" * 4], []]
    assert _records(tmp_path) == []


@pytest.mark.parametrize(
    "spec, match",
    [
        ({"kind": "flip"}, "Unknown KV mutation kind"),
        ({"kind": "zero", "fraction": 0.0}, "must be in"),
        ({"kind": "zero", "fraction": 1.5}, "must be in"),
    ],
)
def test_a_bad_spec_is_rejected(tmp_path: Path, spec: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _install(tmp_path, **spec)


def test_installing_twice_wraps_the_sender_once(tmp_path: Path) -> None:
    module = _module()
    spec = {"kind": "zero", "prompt_len": 130, "record_dir": str(tmp_path / "records")}
    _PROBE.install_sender_mutation(module, spec)
    first = module.Sender._deliver_kv_to_agent
    _PROBE.install_sender_mutation(module, spec)
    assert module.Sender._deliver_kv_to_agent is first
