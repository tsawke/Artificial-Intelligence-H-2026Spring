from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(slots=True)
class SeedSets:
    campaign1: list[int]
    campaign2: list[int]

    @property
    def total_size(self) -> int:
        return len(self.campaign1) + len(self.campaign2)


@dataclass(slots=True)
class GraphData:
    num_nodes: int
    num_edges: int
    out_neighbors: list[list[int]]
    out_prob1: list[list[float]]
    out_prob2: list[list[float]]
    in_neighbors: list[list[int]]
    out_weight_sum1: list[float]
    out_weight_sum2: list[float]


@dataclass(slots=True)
class CommonArgs:
    network_path: str
    initial_seed_path: str
    balanced_seed_path: str
    budget: int
    object_output_path: Optional[str] = None


@dataclass(slots=True)
class WorkBuffers:
    visited: bytearray
    reached: bytearray
    scratch: bytearray
    frontier: list[int]
    queue: list[int]


class Timer:
    __slots__ = ("_start", "_deadline")

    def __init__(self, time_limit_seconds: Optional[float] = None, reserve_seconds: float = 0.0) -> None:
        self._start = time.perf_counter()
        if time_limit_seconds is None:
            self._deadline = None
        else:
            self._deadline = self._start + max(0.0, time_limit_seconds - reserve_seconds)

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self) -> float:
        if self._deadline is None:
            return float("inf")
        return self._deadline - time.perf_counter()

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def has_time(self, seconds_needed: float = 0.0) -> bool:
        return self.remaining() > seconds_needed

    def checkpoint(self) -> None:
        if self.expired():
            raise TimeoutError("Time budget exhausted.")


class RandomContext:
    __slots__ = ("seed", "py_random", "_np_rng")

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.py_random = random.Random(seed)
        self._np_rng = None

    def numpy_rng(self):
        if self._np_rng is None:
            import numpy as np

            self._np_rng = np.random.default_rng(self.seed)
        return self._np_rng


def normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def build_common_arg_parser(needs_output_path: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", required=True, help="Absolute path of the social network file.")
    parser.add_argument("-i", "--initial-seed", required=True, help="Absolute path of the initial seed file.")
    parser.add_argument("-b", "--balanced-seed", required=True, help="Balanced seed path.")
    parser.add_argument("-k", "--budget", required=True, type=int, help="Positive integer budget.")
    if needs_output_path:
        parser.add_argument("-o", "--output", required=True, help="Absolute path of the objective output file.")
    return parser


def parse_common_args(argv: Optional[Sequence[str]] = None, needs_output_path: bool = False) -> CommonArgs:
    parser = build_common_arg_parser(needs_output_path=needs_output_path)
    namespace = parser.parse_args(argv)
    if namespace.budget <= 0:
        raise ValueError("Budget must be a positive integer.")

    return CommonArgs(
        network_path=normalize_path(namespace.network),
        initial_seed_path=normalize_path(namespace.initial_seed),
        balanced_seed_path=normalize_path(namespace.balanced_seed),
        budget=namespace.budget,
        object_output_path=normalize_path(namespace.output) if needs_output_path else None,
    )


def _read_nonempty_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_graph(path: str) -> GraphData:
    lines = _read_nonempty_lines(path)
    if not lines:
        raise ValueError(f"Graph file is empty: {path}")

    first = lines[0].split()
    if len(first) != 2:
        raise ValueError(f"Graph header must contain exactly two integers: {path}")

    num_nodes, declared_edges = map(int, first)
    out_neighbors = [[] for _ in range(num_nodes)]
    out_prob1 = [[] for _ in range(num_nodes)]
    out_prob2 = [[] for _ in range(num_nodes)]
    in_neighbors = [[] for _ in range(num_nodes)]
    out_weight_sum1 = [0.0] * num_nodes
    out_weight_sum2 = [0.0] * num_nodes

    actual_edges = 0
    for raw in lines[1:]:
        parts = raw.split()
        if len(parts) != 4:
            raise ValueError(f"Each edge line must contain four columns: {raw}")
        src, dst = map(int, parts[:2])
        p1 = float(parts[2])
        p2 = float(parts[3])
        if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
            raise ValueError(f"Edge endpoint out of range: {raw}")

        out_neighbors[src].append(dst)
        out_prob1[src].append(p1)
        out_prob2[src].append(p2)
        in_neighbors[dst].append(src)
        out_weight_sum1[src] += p1
        out_weight_sum2[src] += p2
        actual_edges += 1

    if actual_edges != declared_edges:
        raise ValueError(
            f"Edge count mismatch for {path}: declared {declared_edges}, parsed {actual_edges}."
        )

    return GraphData(
        num_nodes=num_nodes,
        num_edges=actual_edges,
        out_neighbors=out_neighbors,
        out_prob1=out_prob1,
        out_prob2=out_prob2,
        in_neighbors=in_neighbors,
        out_weight_sum1=out_weight_sum1,
        out_weight_sum2=out_weight_sum2,
    )


def load_seed_sets(path: str) -> SeedSets:
    lines = _read_nonempty_lines(path)
    if not lines:
        raise ValueError(f"Seed file is empty: {path}")

    first = lines[0].split()
    if len(first) != 2:
        raise ValueError(f"Seed header must contain exactly two integers: {path}")

    k1, k2 = map(int, first)
    expected = 1 + k1 + k2
    if len(lines) != expected:
        raise ValueError(
            f"Seed count mismatch for {path}: expected {expected - 1} seeds, parsed {len(lines) - 1}."
        )

    campaign1 = [int(value) for value in lines[1 : 1 + k1]]
    campaign2 = [int(value) for value in lines[1 + k1 : 1 + k1 + k2]]
    return SeedSets(campaign1=campaign1, campaign2=campaign2)


def validate_seed_sets(seed_sets: SeedSets, num_nodes: Optional[int] = None, budget: Optional[int] = None) -> None:
    if len(set(seed_sets.campaign1)) != len(seed_sets.campaign1):
        raise ValueError("Campaign 1 seed set contains duplicate nodes.")
    if len(set(seed_sets.campaign2)) != len(seed_sets.campaign2):
        raise ValueError("Campaign 2 seed set contains duplicate nodes.")

    if num_nodes is not None:
        for node in seed_sets.campaign1 + seed_sets.campaign2:
            if not 0 <= node < num_nodes:
                raise ValueError(f"Seed node out of range: {node}")

    if budget is not None and seed_sets.total_size > budget:
        raise ValueError(
            f"Budget exceeded: {seed_sets.total_size} seeds provided, budget is {budget}."
        )


def write_seed_sets(path: str, seed_sets: SeedSets) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{len(seed_sets.campaign1)} {len(seed_sets.campaign2)}\n")
        for node in seed_sets.campaign1:
            handle.write(f"{node}\n")
        for node in seed_sets.campaign2:
            handle.write(f"{node}\n")


def write_objective_value(path: str, value: float) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{value}\n")


def create_work_buffers(num_nodes: int) -> WorkBuffers:
    return WorkBuffers(
        visited=bytearray(num_nodes),
        reached=bytearray(num_nodes),
        scratch=bytearray(num_nodes),
        frontier=[],
        queue=[],
    )


Z_95 = 1.959963984540054


@dataclass(slots=True)
class SamplePlan:
    min_samples: int
    max_samples: int
    batch_size: int
    absolute_half_width: float
    relative_half_width: float


class OnlineStats:
    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def half_width_95(self) -> float:
        if self.count < 2:
            return float("inf")
        variance = self.m2 / (self.count - 1)
        return Z_95 * math.sqrt(variance / self.count)


class ByteWorkspace:
    __slots__ = ("active", "reached", "marked", "queue", "touched_active", "touched_reached", "union_nodes")

    def __init__(self, num_nodes: int) -> None:
        self.active = bytearray(num_nodes)
        self.reached = bytearray(num_nodes)
        self.marked = bytearray(num_nodes)
        self.queue: list[int] = []
        self.touched_active: list[int] = []
        self.touched_reached: list[int] = []
        self.union_nodes: list[int] = []

    def reset(self) -> None:
        for node in self.touched_active:
            self.active[node] = 0
        for node in self.touched_reached:
            self.reached[node] = 0
        self.queue.clear()
        self.touched_active.clear()
        self.touched_reached.clear()


def choose_sample_plan(num_nodes: int, num_edges: int) -> SamplePlan:
    if num_nodes <= 1000 and num_edges <= 50000:
        return SamplePlan(
            min_samples=256,
            max_samples=768,
            batch_size=64,
            absolute_half_width=1.5,
            relative_half_width=0.0030,
        )
    if num_edges <= 100000:
        return SamplePlan(
            min_samples=96,
            max_samples=224,
            batch_size=32,
            absolute_half_width=max(10.0, num_nodes * 0.0004),
            relative_half_width=0.0015,
        )
    return SamplePlan(
        min_samples=80,
        max_samples=160,
        batch_size=16,
        absolute_half_width=max(12.0, num_nodes * 0.0005),
        relative_half_width=0.0012,
    )


def merged_seed_list(seed_a: list[int], seed_b: list[int]) -> list[int]:
    return sorted(set(seed_a) | set(seed_b))


def simulate_reached_bits(
    out_neighbors: list[list[int]],
    out_probs: list[list[float]],
    seeds: list[int],
    rng,
) -> int:
    active_bits = 0
    reached_bits = 0
    queue: list[int] = []

    for node in seeds:
        bit = 1 << node
        if active_bits & bit:
            continue
        active_bits |= bit
        reached_bits |= bit
        queue.append(node)

    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1

        neighbors = out_neighbors[source]
        probs = out_probs[source]
        for idx, target in enumerate(neighbors):
            bit = 1 << target
            if active_bits & bit:
                continue

            reached_bits |= bit
            if rng.random() < probs[idx]:
                active_bits |= bit
                queue.append(target)

    return reached_bits


def simulate_reached_bytes(
    out_neighbors: list[list[int]],
    out_probs: list[list[float]],
    seeds: list[int],
    rng,
    workspace: ByteWorkspace,
) -> bytearray:
    workspace.reset()

    active = workspace.active
    reached = workspace.reached
    queue = workspace.queue
    touched_active = workspace.touched_active
    touched_reached = workspace.touched_reached

    for node in seeds:
        if not active[node]:
            active[node] = 1
            touched_active.append(node)
            queue.append(node)
        if not reached[node]:
            reached[node] = 1
            touched_reached.append(node)

    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1

        neighbors = out_neighbors[source]
        probs = out_probs[source]
        for idx, target in enumerate(neighbors):
            if active[target]:
                continue

            if not reached[target]:
                reached[target] = 1
                touched_reached.append(target)
            if rng.random() < probs[idx]:
                active[target] = 1
                touched_active.append(target)
                queue.append(target)

    return reached


def evaluate_small_graph(graph, campaign1_seeds: list[int], campaign2_seeds: list[int], plan: SamplePlan, rng) -> float:
    stats = OnlineStats()
    target_bits = (1 << graph.num_nodes) - 1

    while stats.count < plan.max_samples:
        batch = min(plan.batch_size, plan.max_samples - stats.count)
        for _ in range(batch):
            reached1 = simulate_reached_bits(graph.out_neighbors, graph.out_prob1, campaign1_seeds, rng)
            reached2 = simulate_reached_bits(graph.out_neighbors, graph.out_prob2, campaign2_seeds, rng)
            balanced = graph.num_nodes - ((reached1 ^ reached2) & target_bits).bit_count()
            stats.update(float(balanced))

        if stats.count < plan.min_samples:
            continue

        half_width = stats.half_width_95()
        if half_width <= max(plan.absolute_half_width, stats.mean * plan.relative_half_width):
            break

    return stats.mean


def evaluate_large_graph(graph, campaign1_seeds: list[int], campaign2_seeds: list[int], plan: SamplePlan, rng) -> float:
    stats = OnlineStats()
    workspace1 = ByteWorkspace(graph.num_nodes)
    workspace2 = ByteWorkspace(graph.num_nodes)

    while stats.count < plan.max_samples:
        batch = min(plan.batch_size, plan.max_samples - stats.count)
        for _ in range(batch):
            reached1 = simulate_reached_bytes(graph.out_neighbors, graph.out_prob1, campaign1_seeds, rng, workspace1)
            reached2 = simulate_reached_bytes(graph.out_neighbors, graph.out_prob2, campaign2_seeds, rng, workspace2)

            union_nodes = workspace1.union_nodes
            union_nodes.clear()
            marked = workspace1.marked

            for node in workspace1.touched_reached:
                if not marked[node]:
                    marked[node] = 1
                    union_nodes.append(node)
            for node in workspace2.touched_reached:
                if not marked[node]:
                    marked[node] = 1
                    union_nodes.append(node)

            mismatch = 0
            for node in union_nodes:
                mismatch += reached1[node] ^ reached2[node]
                marked[node] = 0

            balanced = graph.num_nodes - mismatch
            stats.update(float(balanced))

        if stats.count < plan.min_samples:
            continue

        half_width = stats.half_width_95()
        if half_width <= max(plan.absolute_half_width, stats.mean * plan.relative_half_width):
            break

    return stats.mean


def evaluate_objective(graph, initial_seed_sets, balanced_seed_sets, random_seed: int = 20260329) -> float:
    campaign1_seeds = merged_seed_list(initial_seed_sets.campaign1, balanced_seed_sets.campaign1)
    campaign2_seeds = merged_seed_list(initial_seed_sets.campaign2, balanced_seed_sets.campaign2)

    random_context = RandomContext(seed=random_seed)
    plan = choose_sample_plan(graph.num_nodes, graph.num_edges)

    if graph.num_nodes <= 1024:
        return evaluate_small_graph(graph, campaign1_seeds, campaign2_seeds, plan, random_context.py_random)
    return evaluate_large_graph(graph, campaign1_seeds, campaign2_seeds, plan, random_context.py_random)


def main() -> None:
    args = parse_common_args(needs_output_path=True)
    try:
        graph = load_graph(args.network_path)
        initial_seed_sets = load_seed_sets(args.initial_seed_path)
        balanced_seed_sets = load_seed_sets(args.balanced_seed_path)

        validate_seed_sets(initial_seed_sets, num_nodes=graph.num_nodes)
        validate_seed_sets(balanced_seed_sets, num_nodes=graph.num_nodes, budget=args.budget)

        objective_value = evaluate_objective(graph, initial_seed_sets, balanced_seed_sets)
    except Exception:
        objective_value = 0.0
    write_objective_value(args.object_output_path, objective_value)


if __name__ == "__main__":
    main()
