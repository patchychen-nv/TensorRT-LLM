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
"""Deadlock check of a DKV data-plane plan, without a GPU.

A plan is only safe if the operations that the ranks enqueue can all finish. ``simulate`` expands a
plan into the per-rank operation sequences (``expand_plan``), adds the dependencies between the
operations and checks three things:

* pairing: the k-th send from rank a to rank b meets the k-th receive that b posts from a (first in,
  first out per peer), and both describe the same step, deadline class and requests. A different
  size hangs the transport or corrupts the data, and so does the same size with other content;
* order: every rank issues the steps in the global order of ``global_step_order``, which the
  argument for deadlock freedom relies on;
* wait-for cycles: the graph of all the operations and of the compute they interleave with has no
  cycle. A cycle is a deadlock.

The model has two in-order queues per rank.

* The compute stream runs the attention ``A(l)`` and then the MoE ``M(l)`` of every layer, in layer
  order. Every rank runs every layer, a rank that holds only dummy requests too.
* The data stream runs the operations of the rank's program one after another.
* A message is a rendezvous: a send and its receive finish together, and neither can finish before
  the other has started. This is the worst case; a transport that lets small messages finish early
  only removes waits.
* ``A(l)`` also waits for the fetched KV of layer ``l`` on its rank, which is the receives and the
  local gather of ``F(l)``.
* ``M(l)`` waits for ``A(l)`` on every rank, its own included, because the MoE exchange is
  collective.
* An operation that reads the staging slot of layer ``l`` (a send or the local scatter of
  ``W(l)``) waits for ``A(l)`` on its rank, which produces what it reads.
* An operation that writes the staging slot of layer ``m`` (a receive or the local gather of
  ``F(m)``) overwrites what layer ``m - ring_depth`` left there. It waits for ``M(m - ring_depth)``
  on its rank and for the operations of ``W(m - ring_depth)`` on its rank, which read the slot.
* An operation that the hook at the top of layer ``l`` enqueues waits for ``M(l - 1)`` on its rank,
  because the data stream waits for the kernels enqueued before the hook. Every rank does so, one
  that computes nothing as well: a receive that started earlier would spin on its GPU for a message
  that is not sent yet.

Not modeled: the host side of the hooks, which enqueue an operation only after the events it waits
for are recorded and never block while doing it; communicators other than the one of the data plane;
and the sharing of the GPU's multiprocessors between kernels (the wait above only keeps a receive
from spinning for long before its layer).

Usage::

    plan = build_dkv_plan(...)
    report = simulate(plan)
    assert report.ok, str(report)

    programs = expand_plan(plan)  # per-rank lists, which can be edited to try a bad plan
    report = simulate(plan, programs)
"""

import enum
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from .dkv_plan import Direction, DkvPlan, OpAction, RankOp, global_step_order, step_label

__all__ = [
    "Problem",
    "ProblemKind",
    "SimulationReport",
    "expand_plan",
    "simulate",
]

# How many mismatched messages one report lists before it summarizes the rest.
_MAX_LISTED_MISMATCHES = 8
# How many operations of a cycle a message lists.
_MAX_LISTED_CYCLE = 20
# From how many operations of the cycle that was found a shorter one is searched.
_MAX_CYCLE_SEARCHES = 64


class ProblemKind(enum.Enum):
    """What kind of defect a problem is."""

    MALFORMED = "malformed program"
    UNPAIRED = "unpaired message"
    SIZE_MISMATCH = "size mismatch"
    STEP_MISMATCH = "step mismatch"
    STEP_ORDER = "step order violated"
    CYCLE = "wait-for cycle"


@dataclass(frozen=True)
class Problem:
    """One defect found in a set of programs.

    Attributes:
        kind: The kind of defect.
        message: A readable description that names the ranks, layers and steps involved.
    """

    kind: ProblemKind
    message: str


@dataclass(frozen=True)
class SimulationReport:
    """The outcome of ``simulate``.

    Attributes:
        problems: Every defect found; empty if the programs are deadlock free.
        num_ops: Data-stream operations over all ranks.
        num_messages: Messages, each a send paired with a receive.
        num_nodes: Nodes of the wait-for graph (zero if it was not built).
        num_edges: Dependencies of the wait-for graph (zero if it was not built).
        graph_checked: Whether the wait-for graph was built and checked for cycles. It is not if
            the programs are malformed or contain a message without a partner, which already
            hangs.
    """

    problems: tuple[Problem, ...]
    num_ops: int
    num_messages: int
    num_nodes: int
    num_edges: int
    graph_checked: bool

    @property
    def ok(self) -> bool:
        """Whether no defect was found."""
        return not self.problems

    def of_kind(self, kind: ProblemKind) -> tuple[Problem, ...]:
        """The problems of one kind."""
        return tuple(problem for problem in self.problems if problem.kind is kind)

    def __str__(self) -> str:
        summary = (
            f"{self.num_ops} operations, {self.num_messages} messages, "
            f"{self.num_nodes} nodes, {self.num_edges} dependencies"
        )
        if self.ok:
            return f"deadlock free: {summary}"
        lines = [f"{len(self.problems)} problem(s) in {summary}"]
        for problem in self.problems:
            lines.append(f"- {problem.kind.value}: {problem.message}")
        return "\n".join(lines)


def expand_plan(plan: DkvPlan) -> list[list[RankOp]]:
    """The data-stream program of every rank of ``plan``, as lists that can be edited."""
    return [list(plan.rank_program(rank)) for rank in range(plan.group_size)]


def simulate(plan: DkvPlan, programs: Sequence[Sequence[RankOp]] | None = None) -> SimulationReport:
    """Check that the data plane of a plan pairs up and cannot deadlock.

    Args:
        plan: The plan, which gives the number of ranks and layers and the ring depth.
        programs: The operation sequence of every rank. Defaults to ``expand_plan(plan)``. Pass
            edited programs to see how the checks react to a bad plan.

    Returns:
        The report; ``report.ok`` is true if no defect was found.
    """
    if programs is None:
        programs = expand_plan(plan)
    return _Simulation(plan, programs).run()


# A node of the wait-for graph: its kind and up to four integers that say which one it is. The
# kinds are "attention" (rank, layer), "moe" (rank, layer), "exchange" (layer), "message" (sender,
# send op, receiver, receive op) and "local" (rank, op). Unused integers are zero.
_Key = tuple[str, int, int, int, int]
# Why a node waits for another: a code and up to three integers, as in the table of
# ``_describe_reason``.
_Reason = tuple[str, int, int, int]


class _Graph:
    """Nodes that wait for other nodes; the ids count up in creation order."""

    def __init__(self) -> None:
        self.keys: list[_Key] = []
        self._ids: dict[_Key, int] = {}
        # For every node the nodes it waits for, each with the reason.
        self.prerequisites: list[dict[int, _Reason]] = []

    def node(self, key: _Key) -> int:
        node_id = self._ids.get(key)
        if node_id is None:
            node_id = len(self.keys)
            self._ids[key] = node_id
            self.keys.append(key)
            self.prerequisites.append({})
        return node_id

    def wait(self, node: int, prerequisite: int, reason: _Reason) -> None:
        """Record that ``node`` can finish only after ``prerequisite`` finished."""
        self.prerequisites[node].setdefault(prerequisite, reason)

    def num_edges(self) -> int:
        return sum(len(prerequisites) for prerequisites in self.prerequisites)


class _Simulation:
    def __init__(self, plan: DkvPlan, programs: Sequence[Sequence[RankOp]]) -> None:
        self.plan = plan
        self.programs = [tuple(program) for program in programs]
        self.problems: list[Problem] = []
        order = global_step_order(plan.num_layers, plan.ring_depth)
        self.step_index = {(step.direction, step.layer): index for index, step in enumerate(order)}
        self.issue_point = {(step.direction, step.layer): step.issue_point for step in order}
        # (sender, send op index, receiver, receive op index) of every message.
        self.messages: list[tuple[int, int, int, int]] = []
        self.unpaired = False
        self.graph: _Graph | None = None

    def run(self) -> SimulationReport:
        if self._check_programs():
            self._check_step_order()
            self._pair_messages()
            if not self.unpaired:
                self._check_wait_for_graph()
        num_ops = sum(len(program) for program in self.programs)
        graph = self.graph
        return SimulationReport(
            problems=tuple(self.problems),
            num_ops=num_ops,
            num_messages=len(self.messages),
            num_nodes=len(graph.keys) if graph is not None else 0,
            num_edges=graph.num_edges() if graph is not None else 0,
            graph_checked=graph is not None,
        )

    def _problem(self, kind: ProblemKind, message: str) -> None:
        self.problems.append(Problem(kind, message))

    def _check_programs(self) -> bool:
        """Check that the programs can be analyzed at all; report the first defects if not."""
        plan = self.plan
        if len(self.programs) != plan.group_size:
            self._problem(
                ProblemKind.MALFORMED,
                f"the plan has {plan.group_size} ranks but {len(self.programs)} programs "
                "were given",
            )
            return False
        defects = []
        for rank, program in enumerate(self.programs):
            for index, op in enumerate(program):
                transfer = op.transfer
                where = f"rank {rank} op #{index}"
                if (transfer.direction, transfer.layer) not in self.step_index:
                    defects.append(
                        f"{where} belongs to {step_label(transfer.direction, transfer.layer)}, "
                        f"which is not a step of a plan with {plan.num_layers} layers"
                    )
                elif not (
                    0 <= transfer.owner < plan.group_size
                    and 0 <= transfer.compute < plan.group_size
                ):
                    defects.append(f"{where} names a rank outside the group ({op.describe()})")
                elif op.rank != rank:
                    defects.append(f"{where} is an operation of rank {op.rank}: {op.describe()}")
                elif op.action is not OpAction.LOCAL and op.peer == rank:
                    defects.append(f"{where} is a message from the rank to itself")
        for defect in defects[:_MAX_LISTED_MISMATCHES]:
            self._problem(ProblemKind.MALFORMED, defect)
        return not defects

    def _check_step_order(self) -> None:
        """Every rank must issue the steps in the global order."""
        for rank, program in enumerate(self.programs):
            furthest = -1
            furthest_op = 0
            reported: set[tuple[int, int]] = set()
            violations: list[tuple[int, int]] = []
            for index, op in enumerate(program):
                transfer = op.transfer
                position = self.step_index[(transfer.direction, transfer.layer)]
                if position >= furthest:
                    furthest, furthest_op = position, index
                elif (furthest, position) not in reported:
                    reported.add((furthest, position))
                    violations.append((furthest_op, index))
            if not violations:
                continue
            late_index, early_index = violations[0]
            late = program[late_index].transfer
            early = program[early_index].transfer
            message = (
                f"rank {rank} issues {step_label(late.direction, late.layer)} (op #{late_index}) "
                f"before {step_label(early.direction, early.layer)} (op #{early_index}), but the "
                f"global step order puts {step_label(early.direction, early.layer)} first"
            )
            if len(violations) > 1:
                message += f"; {len(violations) - 1} more steps are out of order on this rank"
            self._problem(ProblemKind.STEP_ORDER, message)

    def _pair_messages(self) -> None:
        """Pair the k-th send of a to b with the k-th receive of b from a."""
        sends: dict[tuple[int, int], list[int]] = {}
        receives: dict[tuple[int, int], list[int]] = {}
        for rank, program in enumerate(self.programs):
            for index, op in enumerate(program):
                if op.action is OpAction.SEND:
                    sends.setdefault((rank, op.peer), []).append(index)
                elif op.action is OpAction.RECV:
                    receives.setdefault((op.peer, rank), []).append(index)
        mismatches = 0
        for sender, receiver in sorted(set(sends) | set(receives)):
            send_ops = sends.get((sender, receiver), [])
            receive_ops = receives.get((sender, receiver), [])
            for number, (i, j) in enumerate(zip(send_ops, receive_ops), start=1):
                self.messages.append((sender, i, receiver, j))
                mismatch = self._mismatch(sender, i, receiver, j, number)
                if mismatch is None:
                    continue
                mismatches += 1
                if mismatches <= _MAX_LISTED_MISMATCHES:
                    self._problem(*mismatch)
            if len(send_ops) != len(receive_ops):
                self.unpaired = True
                self._problem(
                    ProblemKind.UNPAIRED, self._unpaired(sender, receiver, send_ops, receive_ops)
                )
        if mismatches > _MAX_LISTED_MISMATCHES:
            self._problem(
                ProblemKind.STEP_MISMATCH,
                f"{mismatches - _MAX_LISTED_MISMATCHES} more mismatched messages are not listed",
            )

    def _mismatch(
        self, sender: int, i: int, receiver: int, j: int, number: int
    ) -> tuple[ProblemKind, str] | None:
        send = self.programs[sender][i]
        receive = self.programs[receiver][j]
        if send.transfer == receive.transfer:
            return None
        where = (
            f"message #{number} from rank {sender} to rank {receiver}: rank {sender} op #{i} is "
            f"'{send.describe()}' but rank {receiver} op #{j} is '{receive.describe()}'"
        )
        if send.nbytes != receive.nbytes:
            return (
                ProblemKind.SIZE_MISMATCH,
                f"{where}; the sizes differ ({send.nbytes} B sent, {receive.nbytes} B expected), "
                "so the transport hangs or corrupts the data",
            )
        return (
            ProblemKind.STEP_MISMATCH,
            f"{where}; the sizes agree but the messages are not the same, so the data would "
            "be silently delivered to the wrong place",
        )

    def _unpaired(
        self, sender: int, receiver: int, send_ops: list[int], receive_ops: list[int]
    ) -> str:
        if len(send_ops) > len(receive_ops):
            extra = send_ops[len(receive_ops) :]
            lacking = (
                f"rank {sender} posts {len(send_ops)} sends to rank {receiver} but rank "
                f"{receiver} posts only {len(receive_ops)} receives from rank {sender}"
            )
            listing = ", ".join(
                f"op #{i} ({self.programs[sender][i].describe()})" for i in extra[:3]
            )
            return f"{lacking}; no receive matches {listing}, so rank {sender} blocks forever"
        extra = receive_ops[len(send_ops) :]
        lacking = (
            f"rank {receiver} posts {len(receive_ops)} receives from rank {sender} but rank "
            f"{sender} posts only {len(send_ops)} sends to rank {receiver}"
        )
        listing = ", ".join(f"op #{j} ({self.programs[receiver][j].describe()})" for j in extra[:3])
        return f"{lacking}; no send matches {listing}, so rank {receiver} waits forever"

    def _check_wait_for_graph(self) -> None:
        graph = self._build_graph()
        self.graph = graph
        stuck = self._stuck_nodes(graph)
        if stuck:
            self._problem(ProblemKind.CYCLE, self._describe_cycle(graph, stuck))

    def _build_graph(self) -> _Graph:
        plan = self.plan
        num_layers, group_size, ring_depth = plan.num_layers, plan.group_size, plan.ring_depth
        graph = _Graph()
        exchange = [graph.node(("exchange", layer, 0, 0, 0)) for layer in range(num_layers)]
        attention = [
            [graph.node(("attention", rank, layer, 0, 0)) for layer in range(num_layers)]
            for rank in range(group_size)
        ]
        moe = [
            [graph.node(("moe", rank, layer, 0, 0)) for layer in range(num_layers)]
            for rank in range(group_size)
        ]
        for layer in range(num_layers):
            for rank in range(group_size):
                graph.wait(
                    exchange[layer], attention[rank][layer], ("exchange needs", rank, layer, 0)
                )
                graph.wait(moe[rank][layer], exchange[layer], ("collective", layer, 0, 0))
                if layer:
                    graph.wait(
                        attention[rank][layer], moe[rank][layer - 1], ("next layer", rank, layer, 0)
                    )

        node_of: dict[tuple[int, int], int] = {}
        for sender, i, receiver, j in self.messages:
            node = graph.node(("message", sender, i, receiver, j))
            node_of[(sender, i)] = node
            node_of[(receiver, j)] = node
        for rank, program in enumerate(self.programs):
            for index in range(len(program)):
                if (rank, index) not in node_of:
                    node_of[(rank, index)] = graph.node(("local", rank, index, 0, 0))

        for rank, program in enumerate(self.programs):
            readers: dict[int, list[int]] = {}
            for index, op in enumerate(program):
                transfer = op.transfer
                if transfer.direction is Direction.WRITEBACK and op.action is not OpAction.RECV:
                    readers.setdefault(transfer.layer, []).append(index)
            for index, op in enumerate(program):
                transfer = op.transfer
                node = node_of[(rank, index)]
                if index:
                    graph.wait(node, node_of[(rank, index - 1)], ("stream", rank, index, 0))
                if transfer.direction is Direction.FETCH:
                    if op.action is not OpAction.SEND:
                        graph.wait(
                            attention[rank][transfer.layer], node, ("needs fetch", rank, index, 0)
                        )
                        earlier = transfer.layer - ring_depth
                        if earlier >= 0:
                            graph.wait(node, moe[rank][earlier], ("slot free", rank, index, 0))
                            for reader in readers.get(earlier, ()):
                                graph.wait(
                                    node,
                                    node_of[(rank, reader)],
                                    ("slot read", rank, index, reader),
                                )
                elif op.action is not OpAction.RECV:
                    graph.wait(node, attention[rank][transfer.layer], ("produced", rank, index, 0))
                issued = self.issue_point[(transfer.direction, transfer.layer)]
                if issued >= 1:
                    graph.wait(node, moe[rank][issued - 1], ("issued after", rank, index, issued))
        return graph

    @staticmethod
    def _stuck_nodes(graph: _Graph) -> list[int]:
        """The nodes that never finish: those on a cycle and those that wait for one."""
        num_nodes = len(graph.keys)
        waiting = [len(prerequisites) for prerequisites in graph.prerequisites]
        dependents: list[list[int]] = [[] for _ in range(num_nodes)]
        for node, prerequisites in enumerate(graph.prerequisites):
            for prerequisite in prerequisites:
                dependents[prerequisite].append(node)
        ready = [node for node in range(num_nodes) if waiting[node] == 0]
        while ready:
            node = ready.pop()
            for dependent in dependents[node]:
                waiting[dependent] -= 1
                if waiting[dependent] == 0:
                    ready.append(dependent)
        return [node for node in range(num_nodes) if waiting[node] > 0]

    def _describe_cycle(self, graph: _Graph, stuck: list[int]) -> str:
        stuck_nodes = set(stuck)
        cycle = self._any_cycle(graph, stuck_nodes, stuck[0])
        # A cycle found by walking is often long; the shortest one through any of its operations
        # is easier to read.
        for member in cycle[:_MAX_CYCLE_SEARCHES]:
            shorter = self._shortest_cycle_through(graph, stuck_nodes, member)
            if shorter and len(shorter) < len(cycle):
                cycle = shorter
        lines = [
            f"deadlock: {len(cycle)} operation(s) wait for each other in a cycle, "
            f"and {len(stuck) - len(cycle)} more wait for them"
        ]
        for k, member in enumerate(cycle[:_MAX_LISTED_CYCLE]):
            following = cycle[(k + 1) % len(cycle)]
            reason = self._describe_reason(graph.prerequisites[member][following])
            lines.append(f"  [{k + 1}] {self._describe_node(graph, member)}")
            lines.append(f"      waits for [{(k + 1) % len(cycle) + 1}] because {reason}")
        if len(cycle) > _MAX_LISTED_CYCLE:
            lines.append(f"  ... {len(cycle) - _MAX_LISTED_CYCLE} more operations in the cycle")
        return "\n".join(lines)

    @staticmethod
    def _any_cycle(graph: _Graph, stuck_nodes: set[int], start: int) -> list[int]:
        """A cycle among the stuck nodes: every stuck node waits for another stuck node."""
        position: dict[int, int] = {}
        path: list[int] = []
        node = start
        while node not in position:
            position[node] = len(path)
            path.append(node)
            node = next(p for p in graph.prerequisites[node] if p in stuck_nodes)
        return path[position[node] :]

    @staticmethod
    def _shortest_cycle_through(graph: _Graph, stuck_nodes: set[int], start: int) -> list[int]:
        """The shortest cycle through ``start``, each node waiting for the next."""
        parent = {start: -1}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for prerequisite in graph.prerequisites[node]:
                if prerequisite not in stuck_nodes:
                    continue
                if prerequisite == start:
                    path = [node]
                    while parent[path[-1]] != -1:
                        path.append(parent[path[-1]])
                    return path[::-1]
                if prerequisite not in parent:
                    parent[prerequisite] = node
                    queue.append(prerequisite)
        return []

    def _describe_node(self, graph: _Graph, node: int) -> str:
        kind, first, second, third, fourth = graph.keys[node]
        if kind == "attention":
            return f"attention of layer {second} on rank {first}"
        if kind == "moe":
            return f"MoE of layer {second} on rank {first}"
        if kind == "exchange":
            return f"start of the MoE exchange of layer {first}"
        if kind == "message":
            transfer = self.programs[first][second].transfer
            return (
                f"message {transfer.label} {transfer.nbytes} B from rank {first} to rank "
                f"{third} (op #{second} on rank {first}, op #{fourth} on rank {third})"
            )
        return f"rank {first} op #{second} ({self.programs[first][second].describe()})"

    @staticmethod
    def _describe_reason(reason: _Reason) -> str:
        code, first, second, third = reason
        if code == "stream":
            return f"rank {first} issues its op #{second} after its previous op"
        if code == "next layer":
            return f"rank {first} computes layer {second} after the layer before"
        if code == "exchange needs":
            return f"the MoE exchange of layer {second} needs the attention of rank {first}"
        if code == "collective":
            return f"the MoE exchange of layer {first} is collective over all ranks"
        if code == "needs fetch":
            return f"attention on rank {first} needs the KV that its op #{second} fetches"
        if code == "slot free":
            return (
                f"op #{second} on rank {first} overwrites the staging slot of an earlier layer, "
                "which must have finished first"
            )
        if code == "slot read":
            return (
                f"op #{second} on rank {first} overwrites a staging slot that its writeback "
                f"op #{third} still has to read"
            )
        if code == "issued after":
            return (
                f"op #{second} on rank {first} is enqueued at the top of layer {third}, behind the "
                f"MoE of the layer before"
            )
        return f"op #{second} on rank {first} sends what the attention produced"
