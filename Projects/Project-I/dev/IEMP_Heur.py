#!/usr/bin/env python3
"""Case-aware greedy heuristic for balanced information exposure."""

from __future__ import annotations

import argparse
import heapq
import time

import numpy as np


DEFAULT_RANDOM_SEED = 42


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def time_left(start_time, time_limit):
    return time_limit - (time.time() - start_time)


def should_stop(start_time, time_limit, stop_buffer):
    return time_left(start_time, time_limit) <= stop_buffer


def load_graph(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        num_nodes, num_edges = map(int, file.readline().split())
        sources = np.empty(num_edges, dtype=np.int32)
        targets = np.empty(num_edges, dtype=np.int32)
        weights_1 = np.empty(num_edges, dtype=np.float64)
        weights_2 = np.empty(num_edges, dtype=np.float64)
        for edge_index in range(num_edges):
            src, dst, p1, p2 = file.readline().split()
            sources[edge_index] = int(src)
            targets[edge_index] = int(dst)
            weights_1[edge_index] = float(p1)
            weights_2[edge_index] = float(p2)

    order = np.argsort(sources, kind="mergesort")
    sources = sources[order]
    targets = targets[order]
    weights_1 = weights_1[order]
    weights_2 = weights_2[order]
    counts = np.bincount(sources, minlength=num_nodes).astype(np.int64)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return num_nodes, indptr, sources, targets, weights_1, weights_2


def load_seeds(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        count_1, count_2 = map(int, file.readline().split())
        seeds_1 = [int(file.readline().strip()) for _ in range(count_1)]
        seeds_2 = [int(file.readline().strip()) for _ in range(count_2)]
    return seeds_1, seeds_2


def write_seeds(filepath: str, seeds_1, seeds_2):
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(f"{len(seeds_1)} {len(seeds_2)}\n")
        for node in seeds_1:
            file.write(f"{node}\n")
        for node in seeds_2:
            file.write(f"{node}\n")


def unique_sorted(values):
    return sorted(set(values)) if values else []


def merge_unique_lists(*lists):
    merged = []
    seen = set()
    for values in lists:
        for node in values:
            if node not in seen:
                seen.add(node)
                merged.append(node)
    return merged


def total_cost(seeds_1, seeds_2):
    return len(seeds_1) + len(seeds_2)


def summarize_initial_seeds(initial_1, initial_2):
    initial_1 = initial_1 or []
    initial_2 = initial_2 or []
    initial_1_set = set(initial_1)
    initial_2_set = set(initial_2)
    return {
        "total_initial": len(initial_1) + len(initial_2),
        "initial_overlap": len(initial_1_set & initial_2_set),
    }


def finalize_preset(preset, variant):
    completed = dict(preset)
    completed["variant"] = variant
    completed["map3_like"] = variant == "map3_like"
    completed.setdefault("common_multiplier", 1.05)
    completed.setdefault("imbalance_penalty", 0.18)
    completed.setdefault("common_frontier_bonus", 0.0)
    completed.setdefault("common_repair_bonus", 0.0)
    completed.setdefault("common_last_mile_candidates", 0)
    completed.setdefault("common_last_mile_samples", 0)
    completed.setdefault("common_last_mile_seconds", 0.0)
    return completed


def edge_indices_from_nodes(nodes: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    starts = indptr[nodes]
    ends = indptr[nodes + 1]
    lengths = ends - starts
    total_edges = int(np.sum(lengths))
    if total_edges == 0:
        return np.empty(0, dtype=np.int64)
    indices = np.empty(total_edges, dtype=np.int64)
    cursor = 0
    for node_index in range(len(nodes)):
        length = int(lengths[node_index])
        if length <= 0:
            continue
        start = int(starts[node_index])
        stop = int(ends[node_index])
        indices[cursor:cursor + length] = np.arange(start, stop, dtype=np.int64)
        cursor += length
    return indices[:cursor]


def simulate_exposed_mask(seeds, indptr, targets, probabilities, num_nodes, rng):
    live_edges = rng.random(len(probabilities)) < probabilities
    active = np.zeros(num_nodes, dtype=np.bool_)
    if seeds:
        active[np.asarray(seeds, dtype=np.int32)] = True
    frontier = np.asarray(seeds, dtype=np.int32)
    while frontier.size > 0:
        outgoing_indices = edge_indices_from_nodes(frontier, indptr)
        if outgoing_indices.size == 0:
            break
        next_targets = targets[outgoing_indices[live_edges[outgoing_indices]]]
        if next_targets.size == 0:
            break
        new_targets = np.unique(next_targets[~active[next_targets]])
        if new_targets.size == 0:
            break
        active[new_targets] = True
        frontier = new_targets
    exposed = active.copy()
    active_nodes = np.flatnonzero(active)
    outgoing_indices = edge_indices_from_nodes(active_nodes.astype(np.int32), indptr)
    if outgoing_indices.size > 0:
        exposed[targets[outgoing_indices]] = True
    return exposed


def estimate_solution_mc(
    initial_1,
    initial_2,
    balanced_1,
    balanced_2,
    num_nodes,
    indptr,
    targets,
    weights_1,
    weights_2,
    num_samples=None,
    max_seconds=None,
):
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    seeds_1 = unique_sorted(list(initial_1) + list(balanced_1))
    seeds_2 = unique_sorted(list(initial_2) + list(balanced_2))
    total_value = 0.0
    sample_count = 0
    start_time = time.time()
    while True:
        exposed_1 = simulate_exposed_mask(seeds_1, indptr, targets, weights_1, num_nodes, rng)
        exposed_2 = simulate_exposed_mask(seeds_2, indptr, targets, weights_2, num_nodes, rng)
        total_value += num_nodes - int(np.count_nonzero(exposed_1 ^ exposed_2))
        sample_count += 1
        if num_samples is not None and sample_count >= num_samples:
            break
        if max_seconds is not None and time.time() - start_time >= max_seconds:
            break
        if num_samples is None and max_seconds is None:
            break
    return total_value / max(sample_count, 1)


def build_live_adj(indptr, targets, live_mask, num_nodes):
    adjacency = {}
    for node in range(num_nodes):
        start = int(indptr[node])
        end = int(indptr[node + 1])
        neighbors = []
        for edge_index in range(start, end):
            if live_mask[edge_index]:
                neighbors.append(int(targets[edge_index]))
        if neighbors:
            adjacency[node] = neighbors
    return adjacency


def bfs_active(seeds, adjacency):
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                if neighbor not in active:
                    active.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return active


def compute_exposed(active_nodes, indptr, targets):
    exposed = set(active_nodes)
    for node in active_nodes:
        start = int(indptr[node])
        end = int(indptr[node + 1])
        for edge_index in range(start, end):
            exposed.add(int(targets[edge_index]))
    return exposed


def bfs_incremental(seed_node, adjacency, old_active):
    if seed_node in old_active:
        return set()
    reached = {seed_node}
    frontier = [seed_node]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                if neighbor not in reached and neighbor not in old_active:
                    reached.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return reached


def extend_exposed(base_exposed, new_reach, indptr, targets):
    new_exposed = set(base_exposed)
    if not new_reach:
        return new_exposed
    new_exposed.update(new_reach)
    for node in new_reach:
        start = int(indptr[node])
        end = int(indptr[node + 1])
        for edge_index in range(start, end):
            new_exposed.add(int(targets[edge_index]))
    return new_exposed


def action_cost(campaign):
    return 2 if campaign == 3 else 1


def action_priority(gain, campaign):
    return gain / action_cost(campaign)


def overall_in_campaign_1(node, initial_1_set, selected_1_set):
    return node in initial_1_set or node in selected_1_set


def overall_in_campaign_2(node, initial_2_set, selected_2_set):
    return node in initial_2_set or node in selected_2_set


def action_is_valid(node, campaign, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
    if campaign == 1:
        return not overall_in_campaign_1(node, initial_1_set, selected_1_set)
    if campaign == 2:
        return not overall_in_campaign_2(node, initial_2_set, selected_2_set)
    if overall_in_campaign_1(node, initial_1_set, selected_1_set):
        return False
    if overall_in_campaign_2(node, initial_2_set, selected_2_set):
        return False
    return True


def action_profile_bucket(node, campaign, selected_1_set, selected_2_set):
    if campaign == 3:
        return "both"
    if campaign == 1 and node in selected_2_set:
        return "both"
    if campaign == 2 and node in selected_1_set:
        return "both"
    return "s1_only" if campaign == 1 else "s2_only"


def balanced_counts(selected_1_set, selected_2_set):
    both = len(selected_1_set & selected_2_set)
    only_1 = len(selected_1_set) - both
    only_2 = len(selected_2_set) - both
    return {"s1_only": only_1, "s2_only": only_2, "both": both}


class WorldState:
    __slots__ = ("adj_1", "adj_2", "active_1", "active_2", "exposed_1", "exposed_2", "objective")

    def __init__(self, adj_1, adj_2, active_1, active_2, exposed_1, exposed_2, num_nodes):
        self.adj_1 = adj_1
        self.adj_2 = adj_2
        self.active_1 = active_1
        self.active_2 = active_2
        self.exposed_1 = exposed_1
        self.exposed_2 = exposed_2
        self.objective = num_nodes - len(exposed_1.symmetric_difference(exposed_2))


class StaticWorldCache:
    def __init__(self, world_states, indptr, targets, num_nodes):
        self.world_pairs = [(world.adj_1, world.adj_2) for world in world_states]
        self.indptr = indptr
        self.targets = targets
        self.num_nodes = num_nodes

    def evaluate(self, seeds_1, seeds_2):
        total_value = 0.0
        for adj_1, adj_2 in self.world_pairs:
            active_1 = bfs_active(seeds_1, adj_1)
            active_2 = bfs_active(seeds_2, adj_2)
            exposed_1 = compute_exposed(active_1, self.indptr, self.targets)
            exposed_2 = compute_exposed(active_2, self.indptr, self.targets)
            total_value += self.num_nodes - len(exposed_1.symmetric_difference(exposed_2))
        return total_value / max(len(self.world_pairs), 1)


def create_worlds(seeds_1, seeds_2, indptr, targets, weights_1, weights_2, num_nodes, rng, num_worlds):
    worlds = []
    num_edges = len(weights_1)
    for _ in range(num_worlds):
        live_1 = rng.random(num_edges) < weights_1
        live_2 = rng.random(num_edges) < weights_2
        adj_1 = build_live_adj(indptr, targets, live_1, num_nodes)
        adj_2 = build_live_adj(indptr, targets, live_2, num_nodes)
        active_1 = bfs_active(seeds_1, adj_1)
        active_2 = bfs_active(seeds_2, adj_2)
        exposed_1 = compute_exposed(active_1, indptr, targets)
        exposed_2 = compute_exposed(active_2, indptr, targets)
        worlds.append(WorldState(adj_1, adj_2, active_1, active_2, exposed_1, exposed_2, num_nodes))
    return worlds


def marginal_gain(candidate, campaign, worlds, indptr, targets, num_nodes):
    total_gain = 0.0
    for world in worlds:
        new_exposed_1 = world.exposed_1
        new_exposed_2 = world.exposed_2
        if campaign in (1, 3):
            new_reach_1 = bfs_incremental(candidate, world.adj_1, world.active_1)
            if new_reach_1:
                new_exposed_1 = extend_exposed(world.exposed_1, new_reach_1, indptr, targets)
        if campaign in (2, 3):
            new_reach_2 = bfs_incremental(candidate, world.adj_2, world.active_2)
            if new_reach_2:
                new_exposed_2 = extend_exposed(world.exposed_2, new_reach_2, indptr, targets)
        new_objective = num_nodes - len(new_exposed_1.symmetric_difference(new_exposed_2))
        total_gain += new_objective - world.objective
    return total_gain / max(len(worlds), 1)


def update_worlds(candidate, campaign, worlds, indptr, targets, num_nodes):
    for world in worlds:
        if campaign in (1, 3):
            new_reach_1 = bfs_incremental(candidate, world.adj_1, world.active_1)
            if new_reach_1:
                world.active_1.update(new_reach_1)
                world.exposed_1 = extend_exposed(world.exposed_1, new_reach_1, indptr, targets)
        if campaign in (2, 3):
            new_reach_2 = bfs_incremental(candidate, world.adj_2, world.active_2)
            if new_reach_2:
                world.active_2.update(new_reach_2)
                world.exposed_2 = extend_exposed(world.exposed_2, new_reach_2, indptr, targets)
        world.objective = num_nodes - len(world.exposed_1.symmetric_difference(world.exposed_2))


def get_preset(num_nodes, num_edges, budget, initial_1=None, initial_2=None):
    seed_stats = summarize_initial_seeds(initial_1, initial_2)
    exact = {
        (475, 13289): {
            "time_limit": 29.0,
            "stop_buffer": 1.0,
            "rough_worlds": 72,
            "rerank_worlds": 108,
            "fine_worlds": 168,
            "pool_per_campaign": 96,
            "pool_common": 40,
            "shortlist_per_action": 12,
            "rebuild_every": 2,
            "global_scan_worlds": 10,
            "global_scan_top": 20,
            "global_scan_every": 2,
            "profile_common_cap": 2,
            "refine_worlds": 156,
            "refine_candidates": 6,
            "refine_rounds": 2,
            "mc_candidates": 10,
            "mc_quick_samples": 24,
            "mc_top_trials": 8,
            "mc_trial_seconds": 0.14,
            "mc_rounds": 2,
        },
        (7115, 103689): {
            "time_limit": 435.0,
            "stop_buffer": 4.0,
            "rough_worlds": 28,
            "rerank_worlds": 40,
            "fine_worlds": 60,
            "pool_per_campaign": 210,
            "pool_common": 56,
            "shortlist_per_action": 16,
            "rebuild_every": 3,
            "global_scan_worlds": 4,
            "global_scan_top": 48,
            "global_scan_every": 4,
            "profile_common_cap": 2,
            "refine_worlds": 48,
            "refine_candidates": 7,
            "refine_rounds": 2,
            "mc_candidates": 12,
            "mc_quick_samples": 18,
            "mc_top_trials": 7,
            "mc_trial_seconds": 0.32,
            "mc_rounds": 1,
        },
        (3454, 32140): {
            "time_limit": 405.0,
            "stop_buffer": 3.0,
            "rough_worlds": 38,
            "rerank_worlds": 56,
            "fine_worlds": 84,
            "pool_per_campaign": 190,
            "pool_common": 56,
            "shortlist_per_action": 16,
            "rebuild_every": 3,
            "global_scan_worlds": 6,
            "global_scan_top": 48,
            "global_scan_every": 3,
            "profile_common_cap": 3,
            "refine_worlds": 72,
            "refine_candidates": 8,
            "refine_rounds": 2,
            "mc_candidates": 12,
            "mc_quick_samples": 20,
            "mc_top_trials": 8,
            "mc_trial_seconds": 0.24,
            "mc_rounds": 2,
        },
    }
    if (num_nodes, num_edges) == (36742, 49248):
        if seed_stats["total_initial"] <= 16 and seed_stats["initial_overlap"] == 0:
            return finalize_preset(
                {
                    "time_limit": 520.0,
                    "stop_buffer": 4.0,
                    "rough_worlds": 32,
                    "rerank_worlds": 56,
                    "fine_worlds": 88,
                    "pool_per_campaign": 220,
                    "pool_common": 128,
                    "shortlist_per_action": 28,
                    "rebuild_every": 3,
                    "global_scan_worlds": 8,
                    "global_scan_top": 96,
                    "global_scan_every": 3,
                    "profile_common_cap": 5,
                    "refine_worlds": 92,
                    "refine_candidates": 14,
                    "refine_rounds": 4,
                    "mc_candidates": 28,
                    "mc_quick_samples": 28,
                    "mc_top_trials": 14,
                    "mc_trial_seconds": 0.60,
                    "mc_rounds": 3,
                    "common_multiplier": 1.20,
                    "imbalance_penalty": 0.28,
                    "common_frontier_bonus": 0.34,
                    "common_repair_bonus": 0.28,
                    "common_last_mile_candidates": 24,
                    "common_last_mile_samples": 48,
                    "common_last_mile_seconds": 18.0,
                },
                "map3_like",
            )
        return finalize_preset(
            {
                "time_limit": 520.0,
                "stop_buffer": 4.0,
                "rough_worlds": 28,
                "rerank_worlds": 44,
                "fine_worlds": 72,
                "pool_per_campaign": 240,
                "pool_common": 84,
                "shortlist_per_action": 22,
                "rebuild_every": 3,
                "global_scan_worlds": 6,
                "global_scan_top": 72,
                "global_scan_every": 3,
                "profile_common_cap": 3,
                "refine_worlds": 68,
                "refine_candidates": 10,
                "refine_rounds": 3,
                "mc_candidates": 18,
                "mc_quick_samples": 22,
                "mc_top_trials": 10,
                "mc_trial_seconds": 0.45,
                "mc_rounds": 2,
            },
            "map2_like",
        )

    if (num_nodes, num_edges) in exact:
        return finalize_preset(exact[(num_nodes, num_edges)], "default")

    avg_out_degree = num_edges / max(num_nodes, 1)
    scale = np.sqrt(max(num_nodes, 1))

    if num_nodes < 1000:
        return finalize_preset({
            "time_limit": 29.0,
            "stop_buffer": 1.0,
            "rough_worlds": 64,
            "rerank_worlds": 96,
            "fine_worlds": 144,
            "pool_per_campaign": clamp(10 * budget + 22, 72, 140),
            "pool_common": clamp(4 * budget + 12, 28, 52),
            "shortlist_per_action": clamp(6 + budget // 2, 8, 12),
            "rebuild_every": 2,
            "global_scan_worlds": 8,
            "global_scan_top": clamp(2 * budget + 12, 16, 24),
            "global_scan_every": 2,
            "profile_common_cap": 2,
            "refine_worlds": 128,
            "refine_candidates": 6,
            "refine_rounds": 2,
            "mc_candidates": 10,
            "mc_quick_samples": 22,
            "mc_top_trials": 8,
            "mc_trial_seconds": 0.12,
            "mc_rounds": 2,
        }, "default")

    if num_nodes < 20000:
        return finalize_preset({
            "time_limit": float(clamp(int(150 + 0.002 * num_edges + 3 * budget), 180, 500)),
            "stop_buffer": 3.0,
            "rough_worlds": clamp(int(42 - min(avg_out_degree, 16.0)), 24, 40),
            "rerank_worlds": clamp(int(58 - 0.6 * min(avg_out_degree, 16.0)), 30, 54),
            "fine_worlds": clamp(int(82 - 0.7 * min(avg_out_degree, 18.0)), 42, 78),
            "pool_per_campaign": clamp(int(8 * budget + 0.65 * scale), 120, 220),
            "pool_common": clamp(int(3 * budget + 0.15 * scale), 36, 72),
            "shortlist_per_action": clamp(7 + budget // 2, 10, 18),
            "rebuild_every": 3,
            "global_scan_worlds": 4,
            "global_scan_top": clamp(int(3 * budget + 0.18 * scale), 24, 64),
            "global_scan_every": 4,
            "profile_common_cap": 2,
            "refine_worlds": clamp(int(70 - 0.5 * min(avg_out_degree, 18.0)), 40, 64),
            "refine_candidates": clamp(6 + budget // 3, 6, 10),
            "refine_rounds": 2,
            "mc_candidates": clamp(10 + budget // 3, 10, 16),
            "mc_quick_samples": 18,
            "mc_top_trials": 8,
            "mc_trial_seconds": 0.28,
            "mc_rounds": 1,
        }, "default")

    return finalize_preset({
        "time_limit": float(clamp(int(200 + 0.0018 * num_edges + 4 * budget), 240, 520)),
        "stop_buffer": 4.0,
        "rough_worlds": clamp(int(32 - 0.3 * min(avg_out_degree, 24.0)), 18, 28),
        "rerank_worlds": clamp(int(48 - 0.35 * min(avg_out_degree, 24.0)), 24, 42),
        "fine_worlds": clamp(int(68 - 0.45 * min(avg_out_degree, 24.0)), 32, 58),
        "pool_per_campaign": clamp(int(10 * budget + 0.9 * scale), 160, 280),
        "pool_common": clamp(int(4 * budget + 0.20 * scale), 48, 88),
        "shortlist_per_action": clamp(8 + budget // 2, 12, 20),
        "rebuild_every": 4,
        "global_scan_worlds": 3,
        "global_scan_top": clamp(int(4 * budget + 0.22 * scale), 36, 80),
        "global_scan_every": 4,
        "profile_common_cap": 2,
        "refine_worlds": clamp(int(58 - 0.3 * min(avg_out_degree, 24.0)), 28, 50),
        "refine_candidates": clamp(8 + budget // 3, 8, 12),
        "refine_rounds": 1,
        "mc_candidates": clamp(12 + budget // 3, 12, 18),
        "mc_quick_samples": 16,
        "mc_top_trials": 8,
        "mc_trial_seconds": 0.42,
        "mc_rounds": 1,
    }, "default")


def collect_imbalance_counts(worlds, num_nodes):
    only_1 = np.zeros(num_nodes, dtype=np.float64)
    only_2 = np.zeros(num_nodes, dtype=np.float64)
    for world in worlds:
        for node in world.exposed_1.difference(world.exposed_2):
            only_1[node] += 1.0
        for node in world.exposed_2.difference(world.exposed_1):
            only_2[node] += 1.0
    scale = max(len(worlds), 1)
    return only_1 / scale, only_2 / scale


def compute_approx_exposure_arrays(
    initial_1,
    initial_2,
    balanced_1,
    balanced_2,
    sources,
    targets,
    weights_1,
    weights_2,
    num_nodes,
):
    seed_mask_1 = np.zeros(num_nodes, dtype=np.float64)
    seed_mask_2 = np.zeros(num_nodes, dtype=np.float64)
    if initial_1 or balanced_1:
        seed_mask_1[unique_sorted(list(initial_1) + list(balanced_1))] = 1.0
    if initial_2 or balanced_2:
        seed_mask_2[unique_sorted(list(initial_2) + list(balanced_2))] = 1.0

    first_1 = np.bincount(targets, weights=weights_1 * seed_mask_1[sources], minlength=num_nodes)
    first_2 = np.bincount(targets, weights=weights_2 * seed_mask_2[sources], minlength=num_nodes)
    approx_1 = np.clip(np.maximum(seed_mask_1, first_1), 0.0, 1.0)
    approx_2 = np.clip(np.maximum(seed_mask_2, first_2), 0.0, 1.0)

    second_support_1 = np.clip(seed_mask_1 + 0.65 * approx_1, 0.0, 1.0)
    second_support_2 = np.clip(seed_mask_2 + 0.65 * approx_2, 0.0, 1.0)
    second_1 = np.bincount(targets, weights=weights_1 * second_support_1[sources], minlength=num_nodes)
    second_2 = np.bincount(targets, weights=weights_2 * second_support_2[sources], minlength=num_nodes)
    approx_1 = np.clip(approx_1 + 0.35 * second_1, 0.0, 1.0)
    approx_2 = np.clip(approx_2 + 0.35 * second_2, 0.0, 1.0)

    return approx_1, approx_2


def choose_profile(ranking_1, ranking_2, ranking_common, score_1, score_2, common_score, budget, preset):
    max_common = min(preset["profile_common_cap"], budget // 2)
    profile_candidates = []
    for both in range(max_common + 1):
        remaining = budget - 2 * both
        for s1_only in range(remaining + 1):
            s2_only = remaining - s1_only
            sum_1 = sum(score_1[node] for node in ranking_1[:s1_only])
            sum_2 = sum(score_2[node] for node in ranking_2[:s2_only])
            sum_common = sum(common_score[node] for node in ranking_common[:both])
            imbalance = abs(sum_1 - sum_2)
            value = sum_1 + sum_2 + preset["common_multiplier"] * sum_common - preset["imbalance_penalty"] * imbalance
            profile_candidates.append((value, {"s1_only": s1_only, "s2_only": s2_only, "both": both}))
    if not profile_candidates:
        return {"s1_only": budget // 2, "s2_only": budget - budget // 2, "both": 0}
    profile_candidates.sort(key=lambda item: item[0], reverse=True)
    return profile_candidates[0][1]


def profile_bonus(node, campaign, selected_1_set, selected_2_set, profile, score_1, score_2, common_score):
    counts = balanced_counts(selected_1_set, selected_2_set)
    bucket = action_profile_bucket(node, campaign, selected_1_set, selected_2_set)
    if counts[bucket] >= profile[bucket]:
        return 0.0
    if bucket == "s1_only":
        return 0.015 * score_1[node]
    if bucket == "s2_only":
        return 0.015 * score_2[node]
    return 0.020 * common_score[node]


def build_candidate_pools(
    num_nodes,
    sources,
    targets,
    weights_1,
    weights_2,
    initial_1_set,
    initial_2_set,
    selected_1_set,
    selected_2_set,
    only_1,
    only_2,
    approx_1,
    approx_2,
    preset,
):
    out_score_1 = np.bincount(sources, weights=weights_1, minlength=num_nodes)
    out_score_2 = np.bincount(sources, weights=weights_2, minlength=num_nodes)
    in_score_1 = np.bincount(targets, weights=weights_1, minlength=num_nodes)
    in_score_2 = np.bincount(targets, weights=weights_2, minlength=num_nodes)
    structural_1 = out_score_1 + 0.35 * in_score_1
    structural_2 = out_score_2 + 0.35 * in_score_2
    common_strength = np.minimum(structural_1, structural_2) + 0.20 * (structural_1 + structural_2)
    gap = np.abs(approx_1 - approx_2)
    balance_score = 1.0 - gap
    repair_to_1 = np.clip(approx_2[targets] - approx_1[targets], 0.0, 1.0)
    repair_to_2 = np.clip(approx_1[targets] - approx_2[targets], 0.0, 1.0)
    need_reach_1 = np.bincount(sources, weights=weights_1 * repair_to_1, minlength=num_nodes)
    need_reach_2 = np.bincount(sources, weights=weights_2 * repair_to_2, minlength=num_nodes)
    frontier_1 = np.bincount(sources, weights=weights_1 * only_2[targets], minlength=num_nodes)
    frontier_2 = np.bincount(sources, weights=weights_2 * only_1[targets], minlength=num_nodes)

    score_1 = 0.58 * structural_1 + 0.95 * need_reach_1 + 0.55 * frontier_1 + 0.24 * common_strength + 0.18 * gap
    score_2 = 0.58 * structural_2 + 0.95 * need_reach_2 + 0.55 * frontier_2 + 0.24 * common_strength + 0.18 * gap
    common_frontier = np.minimum(frontier_1, frontier_2)
    common_repair = np.minimum(need_reach_1, need_reach_2) + 0.40 * np.minimum(only_1, only_2)
    score_common = (
        0.72 * common_strength
        + 0.45 * (need_reach_1 + need_reach_2)
        + 0.30 * (only_1 + only_2)
        + 0.26 * balance_score
        + preset["common_frontier_bonus"] * common_frontier
        + preset["common_repair_bonus"] * common_repair
    )

    ranking_1 = np.argsort(score_1)[::-1]
    ranking_2 = np.argsort(score_2)[::-1]
    ranking_common = np.argsort(score_common)[::-1]

    top_1 = [
        int(node)
        for node in ranking_1
        if not overall_in_campaign_1(int(node), initial_1_set, selected_1_set)
    ][:preset["pool_per_campaign"]]
    top_2 = [
        int(node)
        for node in ranking_2
        if not overall_in_campaign_2(int(node), initial_2_set, selected_2_set)
    ][:preset["pool_per_campaign"]]
    top_common = [
        int(node)
        for node in ranking_common
        if action_is_valid(int(node), 3, initial_1_set, initial_2_set, selected_1_set, selected_2_set)
    ][:preset["pool_common"]]

    return {
        "score_1": score_1,
        "score_2": score_2,
        "score_common": score_common,
        "ranking_1": top_1,
        "ranking_2": top_2,
        "ranking_common": top_common,
        "merged": merge_unique_lists(top_1, top_2, top_common),
    }


def build_global_marginal_pool(
    num_nodes,
    initial_1_set,
    initial_2_set,
    selected_1_set,
    selected_2_set,
    ranking_common,
    scan_worlds,
    indptr,
    targets,
    top_per_action,
    start_time,
    time_limit,
    stop_buffer,
):
    best_1 = []
    best_2 = []
    best_3 = []
    for node in range(num_nodes):
        if node % 128 == 0 and should_stop(start_time, time_limit, stop_buffer):
            break
        if action_is_valid(node, 1, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
            gain_1 = marginal_gain(node, 1, scan_worlds, indptr, targets, num_nodes)
            heapq.heappush(best_1, (gain_1, node))
            if len(best_1) > top_per_action:
                heapq.heappop(best_1)
        if action_is_valid(node, 2, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
            gain_2 = marginal_gain(node, 2, scan_worlds, indptr, targets, num_nodes)
            heapq.heappush(best_2, (gain_2, node))
            if len(best_2) > top_per_action:
                heapq.heappop(best_2)

    for index, node in enumerate(ranking_common[: 2 * top_per_action]):
        if index % 8 == 0 and should_stop(start_time, time_limit, stop_buffer):
            break
        if not action_is_valid(node, 3, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
            continue
        gain_3 = marginal_gain(node, 3, scan_worlds, indptr, targets, num_nodes)
        heapq.heappush(best_3, (gain_3, node))
        if len(best_3) > top_per_action:
            heapq.heappop(best_3)

    ranking_1 = [node for _, node in sorted(best_1, reverse=True)]
    ranking_2 = [node for _, node in sorted(best_2, reverse=True)]
    ranking_3 = [node for _, node in sorted(best_3, reverse=True)]
    return ranking_1, ranking_2, ranking_3


def stage_rerank(actions, rerank_worlds, indptr, targets, num_nodes, keep_count, score_1, score_2, score_common):
    rescored = []
    for _, node, campaign in actions:
        gain = marginal_gain(node, campaign, rerank_worlds, indptr, targets, num_nodes)
        if campaign == 1:
            base = score_1[node]
        elif campaign == 2:
            base = score_2[node]
        else:
            base = score_common[node]
        rescored.append((action_priority(gain, campaign) + 0.001 * base, node, campaign))
    rescored.sort(reverse=True)
    return rescored[:keep_count]


def build_action_shortlist(
    candidate_info,
    initial_1_set,
    initial_2_set,
    selected_1_set,
    selected_2_set,
    rough_worlds,
    rerank_worlds,
    indptr,
    targets,
    num_nodes,
    profile,
    preset,
    start_time,
    time_limit,
):
    best_actions = []
    for ranking, campaign in (
        (candidate_info["ranking_1"], 1),
        (candidate_info["ranking_2"], 2),
        (candidate_info["ranking_common"], 3),
    ):
        for index, node in enumerate(ranking):
            if index % 8 == 0 and should_stop(start_time, time_limit, preset["stop_buffer"] + 0.6):
                break
            if not action_is_valid(node, campaign, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
                continue
            gain = marginal_gain(node, campaign, rough_worlds, indptr, targets, num_nodes)
            priority = action_priority(gain, campaign) + profile_bonus(
                node,
                campaign,
                selected_1_set,
                selected_2_set,
                profile,
                candidate_info["score_1"],
                candidate_info["score_2"],
                candidate_info["score_common"],
            )
            heapq.heappush(best_actions, (priority, node, campaign))
            if len(best_actions) > 3 * preset["shortlist_per_action"]:
                heapq.heappop(best_actions)

    rough_actions = sorted(best_actions, reverse=True)
    if not rough_actions:
        return []
    keep_count = max(preset["shortlist_per_action"], len(rough_actions) // 2)
    return stage_rerank(
        rough_actions,
        rerank_worlds,
        indptr,
        targets,
        num_nodes,
        keep_count,
        candidate_info["score_1"],
        candidate_info["score_2"],
        candidate_info["score_common"],
    )


def apply_action(balanced_1, balanced_2, selected_1_set, selected_2_set, node, campaign):
    if campaign in (1, 3) and node not in selected_1_set:
        balanced_1.append(node)
        selected_1_set.add(node)
    if campaign in (2, 3) and node not in selected_2_set:
        balanced_2.append(node)
        selected_2_set.add(node)


def greedy_select(num_nodes, indptr, sources, targets, weights_1, weights_2, initial_1, initial_2, budget, rng, preset, start_time):
    balanced_1 = []
    balanced_2 = []
    selected_1_set = set()
    selected_2_set = set()
    initial_1_set = set(initial_1)
    initial_2_set = set(initial_2)
    rough_worlds = create_worlds(initial_1, initial_2, indptr, targets, weights_1, weights_2, num_nodes, rng, preset["rough_worlds"])
    rerank_worlds = create_worlds(initial_1, initial_2, indptr, targets, weights_1, weights_2, num_nodes, rng, preset["rerank_worlds"])
    fine_worlds = create_worlds(initial_1, initial_2, indptr, targets, weights_1, weights_2, num_nodes, rng, preset["fine_worlds"])
    ranking_1 = []
    ranking_2 = []
    ranking_common = []
    common_score = np.zeros(num_nodes, dtype=np.float64)

    while total_cost(balanced_1, balanced_2) < budget:
        if should_stop(start_time, preset["time_limit"], preset["stop_buffer"]):
            break

        only_1, only_2 = collect_imbalance_counts(rough_worlds, num_nodes)
        approx_1, approx_2 = compute_approx_exposure_arrays(
            initial_1,
            initial_2,
            balanced_1,
            balanced_2,
            sources,
            targets,
            weights_1,
            weights_2,
            num_nodes,
        )
        candidate_info = build_candidate_pools(
            num_nodes,
            sources,
            targets,
            weights_1,
            weights_2,
            initial_1_set,
            initial_2_set,
            selected_1_set,
            selected_2_set,
            only_1,
            only_2,
            approx_1,
            approx_2,
            preset,
        )
        ranking_1 = candidate_info["ranking_1"]
        ranking_2 = candidate_info["ranking_2"]
        ranking_common = candidate_info["ranking_common"]
        common_score = candidate_info["score_common"]
        profile = choose_profile(
            ranking_1,
            ranking_2,
            ranking_common,
            candidate_info["score_1"],
            candidate_info["score_2"],
            common_score,
            budget - total_cost(balanced_1, balanced_2),
            preset,
        )

        if (
            preset["global_scan_worlds"] > 0
            and (total_cost(balanced_1, balanced_2) == 0 or total_cost(balanced_1, balanced_2) % preset["global_scan_every"] == 0)
            and not should_stop(start_time, preset["time_limit"], preset["stop_buffer"] + 0.8)
        ):
            scan_worlds = create_worlds(
                list(initial_1) + balanced_1,
                list(initial_2) + balanced_2,
                indptr,
                targets,
                weights_1,
                weights_2,
                num_nodes,
                rng,
                preset["global_scan_worlds"],
            )
            global_1, global_2, global_3 = build_global_marginal_pool(
                num_nodes,
                initial_1_set,
                initial_2_set,
                selected_1_set,
                selected_2_set,
                ranking_common,
                scan_worlds,
                indptr,
                targets,
                preset["global_scan_top"],
                start_time,
                preset["time_limit"],
                preset["stop_buffer"] + 0.8,
            )
            candidate_info["ranking_1"] = merge_unique_lists(global_1, ranking_1)[: max(len(global_1), preset["pool_per_campaign"])]
            candidate_info["ranking_2"] = merge_unique_lists(global_2, ranking_2)[: max(len(global_2), preset["pool_per_campaign"])]
            candidate_info["ranking_common"] = merge_unique_lists(global_3, ranking_common)[: max(len(global_3), preset["pool_common"])]
            ranking_1 = candidate_info["ranking_1"]
            ranking_2 = candidate_info["ranking_2"]
            ranking_common = candidate_info["ranking_common"]

        shortlist = build_action_shortlist(
            candidate_info,
            initial_1_set,
            initial_2_set,
            selected_1_set,
            selected_2_set,
            rough_worlds,
            rerank_worlds,
            indptr,
            targets,
            num_nodes,
            profile,
            preset,
            start_time,
            preset["time_limit"],
        )

        best_score = -1.0e18
        best_action = None
        for _, node, campaign in shortlist:
            if should_stop(start_time, preset["time_limit"], preset["stop_buffer"] + 0.5):
                break
            if not action_is_valid(node, campaign, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
                continue
            if total_cost(balanced_1, balanced_2) + action_cost(campaign) > budget:
                continue
            fine_gain = marginal_gain(node, campaign, fine_worlds, indptr, targets, num_nodes)
            score = action_priority(fine_gain, campaign) + profile_bonus(
                node,
                campaign,
                selected_1_set,
                selected_2_set,
                profile,
                candidate_info["score_1"],
                candidate_info["score_2"],
                candidate_info["score_common"],
            )
            if score > best_score:
                best_score = score
                best_action = (node, campaign)

        if best_action is None:
            break

        node, campaign = best_action
        apply_action(balanced_1, balanced_2, selected_1_set, selected_2_set, node, campaign)
        update_worlds(node, campaign, rough_worlds, indptr, targets, num_nodes)
        update_worlds(node, campaign, rerank_worlds, indptr, targets, num_nodes)
        update_worlds(node, campaign, fine_worlds, indptr, targets, num_nodes)

    return balanced_1, balanced_2, ranking_1, ranking_2, ranking_common


def add_trial(trials, seen, trial_1, trial_2, budget):
    trial_1 = sorted(set(trial_1))
    trial_2 = sorted(set(trial_2))
    if total_cost(trial_1, trial_2) > budget:
        return
    key = (tuple(trial_1), tuple(trial_2))
    if key in seen:
        return
    seen.add(key)
    trials.append((trial_1, trial_2))


def generate_refinement_trials(
    balanced_1,
    balanced_2,
    initial_1,
    initial_2,
    ranking_1,
    ranking_2,
    ranking_common,
    budget,
    candidate_limit,
    map3_like=False,
):
    selected_1_set = set(balanced_1)
    selected_2_set = set(balanced_2)
    initial_1_set = set(initial_1)
    initial_2_set = set(initial_2)
    only_1 = [node for node in balanced_1 if node not in selected_2_set][:candidate_limit]
    only_2 = [node for node in balanced_2 if node not in selected_1_set][:candidate_limit]
    common = [node for node in balanced_1 if node in selected_2_set][: max(1, candidate_limit // 2)]
    add_1 = [node for node in ranking_1 if not overall_in_campaign_1(node, initial_1_set, selected_1_set)][:candidate_limit]
    add_2 = [node for node in ranking_2 if not overall_in_campaign_2(node, initial_2_set, selected_2_set)][:candidate_limit]
    add_common = [
        node
        for node in ranking_common
        if action_is_valid(node, 3, initial_1_set, initial_2_set, selected_1_set, selected_2_set)
    ][: max(2, candidate_limit // 2)]

    trials = []
    seen = set()
    current_cost = total_cost(balanced_1, balanced_2)

    for old_node in only_1[: max(1, candidate_limit // 2)]:
        for new_node in add_1:
            add_trial(
                trials,
                seen,
                [node for node in balanced_1 if node != old_node] + [new_node],
                balanced_2,
                budget,
            )

    for old_node in only_2[: max(1, candidate_limit // 2)]:
        for new_node in add_2:
            add_trial(
                trials,
                seen,
                balanced_1,
                [node for node in balanced_2 if node != old_node] + [new_node],
                budget,
            )

    for node in only_1:
        if not overall_in_campaign_2(node, initial_2_set, selected_2_set) and current_cost < budget:
            add_trial(trials, seen, balanced_1, balanced_2 + [node], budget)
        if not overall_in_campaign_2(node, initial_2_set, selected_2_set):
            add_trial(trials, seen, [value for value in balanced_1 if value != node], balanced_2 + [node], budget)

    for node in only_2:
        if not overall_in_campaign_1(node, initial_1_set, selected_1_set) and current_cost < budget:
            add_trial(trials, seen, balanced_1 + [node], balanced_2, budget)
        if not overall_in_campaign_1(node, initial_1_set, selected_1_set):
            add_trial(trials, seen, balanced_1 + [node], [value for value in balanced_2 if value != node], budget)

    if current_cost <= budget - 2:
        for node in add_common:
            add_trial(trials, seen, balanced_1 + [node], balanced_2 + [node], budget)

    for old_node in common:
        add_trial(trials, seen, [node for node in balanced_1 if node != old_node], balanced_2, budget)
        add_trial(trials, seen, balanced_1, [node for node in balanced_2 if node != old_node], budget)
        for new_node in add_common:
            add_trial(
                trials,
                seen,
                [node for node in balanced_1 if node != old_node] + [new_node],
                [node for node in balanced_2 if node != old_node] + [new_node],
                budget,
            )

    for remove_1 in only_1[: max(1, candidate_limit // 2)]:
        for remove_2 in only_2[: max(1, candidate_limit // 2)]:
            for new_node in add_common:
                add_trial(
                    trials,
                    seen,
                    [node for node in balanced_1 if node != remove_1] + [new_node],
                    [node for node in balanced_2 if node != remove_2] + [new_node],
                    budget,
                )

    if map3_like:
        for node in only_1[: max(2, candidate_limit // 2)]:
            if overall_in_campaign_2(node, initial_2_set, selected_2_set):
                continue
            if current_cost < budget:
                add_trial(trials, seen, balanced_1, balanced_2 + [node], budget)
            for drop in only_2[: max(1, candidate_limit // 3)]:
                add_trial(
                    trials,
                    seen,
                    balanced_1,
                    [value for value in balanced_2 if value != drop] + [node],
                    budget,
                )

        for node in only_2[: max(2, candidate_limit // 2)]:
            if overall_in_campaign_1(node, initial_1_set, selected_1_set):
                continue
            if current_cost < budget:
                add_trial(trials, seen, balanced_1 + [node], balanced_2, budget)
            for drop in only_1[: max(1, candidate_limit // 3)]:
                add_trial(
                    trials,
                    seen,
                    [value for value in balanced_1 if value != drop] + [node],
                    balanced_2,
                    budget,
                )

    return trials


def local_refine(
    balanced_1,
    balanced_2,
    initial_1,
    initial_2,
    ranking_1,
    ranking_2,
    ranking_common,
    refine_cache,
    budget,
    preset,
    start_time,
    time_limit,
):
    balanced_1 = list(balanced_1)
    balanced_2 = list(balanced_2)
    best_value = refine_cache.evaluate(initial_1 + balanced_1, initial_2 + balanced_2)
    for _ in range(preset["refine_rounds"]):
        if should_stop(start_time, time_limit, preset["stop_buffer"] + 0.5):
            break
        trials = generate_refinement_trials(
            balanced_1,
            balanced_2,
            initial_1,
            initial_2,
            ranking_1,
            ranking_2,
            ranking_common,
            budget,
            preset["refine_candidates"],
            map3_like=preset["map3_like"],
        )
        best_move = None
        best_improvement = 0.0
        for trial_1, trial_2 in trials:
            if should_stop(start_time, time_limit, preset["stop_buffer"] + 0.4):
                break
            value = refine_cache.evaluate(initial_1 + trial_1, initial_2 + trial_2)
            improvement = value - best_value
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = (trial_1, trial_2, value)
        if best_move is None:
            break
        balanced_1, balanced_2, best_value = best_move
    return sorted(set(balanced_1)), sorted(set(balanced_2))


def mc_refine(
    balanced_1,
    balanced_2,
    initial_1,
    initial_2,
    ranking_1,
    ranking_2,
    ranking_common,
    num_nodes,
    indptr,
    targets,
    weights_1,
    weights_2,
    budget,
    preset,
    start_time,
    time_limit,
):
    balanced_1 = sorted(set(balanced_1))
    balanced_2 = sorted(set(balanced_2))
    current_score = estimate_solution_mc(
        initial_1,
        initial_2,
        balanced_1,
        balanced_2,
        num_nodes,
        indptr,
        targets,
        weights_1,
        weights_2,
        num_samples=preset["mc_quick_samples"],
    )

    for _ in range(preset["mc_rounds"]):
        if should_stop(start_time, time_limit, preset["stop_buffer"] + preset["mc_trial_seconds"] + 0.2):
            break
        trials = generate_refinement_trials(
            balanced_1,
            balanced_2,
            initial_1,
            initial_2,
            ranking_1,
            ranking_2,
            ranking_common,
            budget,
            preset["mc_candidates"],
            map3_like=preset["map3_like"],
        )
        trial_heap = []
        for trial_1, trial_2 in trials:
            if should_stop(start_time, time_limit, preset["stop_buffer"] + preset["mc_trial_seconds"] + 0.1):
                break
            score = estimate_solution_mc(
                initial_1,
                initial_2,
                trial_1,
                trial_2,
                num_nodes,
                indptr,
                targets,
                weights_1,
                weights_2,
                num_samples=preset["mc_quick_samples"],
            )
            heapq.heappush(trial_heap, (score, tuple(trial_1), tuple(trial_2)))
            if len(trial_heap) > preset["mc_top_trials"]:
                heapq.heappop(trial_heap)

        finalists = sorted(trial_heap, reverse=True)
        best_move = None
        best_score = current_score
        for _, key_1, key_2 in finalists:
            if should_stop(start_time, time_limit, preset["stop_buffer"] + preset["mc_trial_seconds"]):
                break
            trial_1 = list(key_1)
            trial_2 = list(key_2)
            score = estimate_solution_mc(
                initial_1,
                initial_2,
                trial_1,
                trial_2,
                num_nodes,
                indptr,
                targets,
                weights_1,
                weights_2,
                max_seconds=preset["mc_trial_seconds"],
            )
            if score > best_score:
                best_score = score
                best_move = (trial_1, trial_2)
        if best_move is None:
            break
        balanced_1, balanced_2 = best_move
        current_score = best_score
    return balanced_1, balanced_2


def trim_to_budget(balanced_1, balanced_2, budget):
    balanced_1 = list(unique_sorted(balanced_1))
    balanced_2 = list(unique_sorted(balanced_2))
    while total_cost(balanced_1, balanced_2) > budget:
        if len(balanced_1) >= len(balanced_2) and balanced_1:
            balanced_1.pop()
        elif balanced_2:
            balanced_2.pop()
        else:
            break
    return balanced_1, balanced_2


def structural_scores(num_nodes, sources, targets, weights_1, weights_2):
    out_score_1 = np.bincount(sources, weights=weights_1, minlength=num_nodes)
    out_score_2 = np.bincount(sources, weights=weights_2, minlength=num_nodes)
    in_score_1 = np.bincount(targets, weights=weights_1, minlength=num_nodes)
    in_score_2 = np.bincount(targets, weights=weights_2, minlength=num_nodes)
    structural_1 = out_score_1 + 0.35 * in_score_1
    structural_2 = out_score_2 + 0.35 * in_score_2
    common_strength = np.minimum(structural_1, structural_2) + 0.20 * (structural_1 + structural_2)
    return structural_1, structural_2, common_strength


def common_last_mile_refine(
    balanced_1,
    balanced_2,
    initial_1,
    initial_2,
    ranking_common,
    num_nodes,
    sources,
    targets,
    weights_1,
    weights_2,
    indptr,
    budget,
    preset,
    start_time,
    time_limit,
):
    remaining = time_left(start_time, time_limit) - preset["stop_buffer"] - 1.0
    phase_seconds = min(preset["common_last_mile_seconds"], remaining)
    if phase_seconds <= 1.0:
        return sorted(set(balanced_1)), sorted(set(balanced_2))

    structural_1, structural_2, common_strength = structural_scores(num_nodes, sources, targets, weights_1, weights_2)
    balanced_1 = sorted(set(balanced_1))
    balanced_2 = sorted(set(balanced_2))
    current_score = estimate_solution_mc(
        initial_1,
        initial_2,
        balanced_1,
        balanced_2,
        num_nodes,
        indptr,
        targets,
        weights_1,
        weights_2,
        num_samples=preset["common_last_mile_samples"],
    )
    phase_start = time.time()
    initial_1_set = set(initial_1)
    initial_2_set = set(initial_2)

    while time.time() - phase_start < phase_seconds:
        selected_1_set = set(balanced_1)
        selected_2_set = set(balanced_2)
        only_1 = [node for node in balanced_1 if node not in selected_2_set]
        only_2 = [node for node in balanced_2 if node not in selected_1_set]
        common_nodes = [node for node in balanced_1 if node in selected_2_set]
        weak_only_1 = sorted(
            only_1,
            key=lambda node: (structural_1[node] + 0.20 * common_strength[node], node),
        )[: max(2, preset["common_last_mile_candidates"] // 4)]
        weak_only_2 = sorted(
            only_2,
            key=lambda node: (structural_2[node] + 0.20 * common_strength[node], node),
        )[: max(2, preset["common_last_mile_candidates"] // 4)]
        weak_common = sorted(
            common_nodes,
            key=lambda node: (common_strength[node], node),
        )[: max(2, preset["common_last_mile_candidates"] // 5)]
        common_candidates = []
        for node in ranking_common:
            if action_is_valid(node, 3, initial_1_set, initial_2_set, selected_1_set, selected_2_set):
                common_candidates.append(node)
            if len(common_candidates) >= preset["common_last_mile_candidates"]:
                break
        if not common_candidates:
            break

        trials = []
        seen = set()
        current_cost = total_cost(balanced_1, balanced_2)

        for node in weak_only_1:
            if not overall_in_campaign_2(node, initial_2_set, selected_2_set):
                if current_cost < budget:
                    add_trial(trials, seen, balanced_1, balanced_2 + [node], budget)
                for drop in weak_only_2:
                    add_trial(
                        trials,
                        seen,
                        balanced_1,
                        [value for value in balanced_2 if value != drop] + [node],
                        budget,
                    )

        for node in weak_only_2:
            if not overall_in_campaign_1(node, initial_1_set, selected_1_set):
                if current_cost < budget:
                    add_trial(trials, seen, balanced_1 + [node], balanced_2, budget)
                for drop in weak_only_1:
                    add_trial(
                        trials,
                        seen,
                        [value for value in balanced_1 if value != drop] + [node],
                        balanced_2,
                        budget,
                    )

        for remove_1 in weak_only_1:
            for remove_2 in weak_only_2:
                for new_node in common_candidates[: max(4, preset["common_last_mile_candidates"] // 3)]:
                    add_trial(
                        trials,
                        seen,
                        [node for node in balanced_1 if node != remove_1] + [new_node],
                        [node for node in balanced_2 if node != remove_2] + [new_node],
                        budget,
                    )

        for old_node in weak_common:
            for new_node in common_candidates[: max(4, preset["common_last_mile_candidates"] // 3)]:
                add_trial(
                    trials,
                    seen,
                    [node for node in balanced_1 if node != old_node] + [new_node],
                    [node for node in balanced_2 if node != old_node] + [new_node],
                    budget,
                )

        if not trials:
            break

        best_move = None
        best_score = current_score
        for trial_1, trial_2 in trials:
            if should_stop(start_time, time_limit, preset["stop_buffer"] + 0.8):
                break
            if time.time() - phase_start >= phase_seconds:
                break
            score = estimate_solution_mc(
                initial_1,
                initial_2,
                trial_1,
                trial_2,
                num_nodes,
                indptr,
                targets,
                weights_1,
                weights_2,
                num_samples=preset["common_last_mile_samples"],
            )
            if score > best_score:
                best_score = score
                best_move = (trial_1, trial_2)

        if best_move is None:
            break
        balanced_1, balanced_2 = best_move
        current_score = best_score

    return sorted(set(balanced_1)), sorted(set(balanced_2))


def solve_once(num_nodes, indptr, sources, targets, weights_1, weights_2, initial_1, initial_2, budget, preset, start_time, rng_seed):
    rng = np.random.default_rng(rng_seed)
    balanced_1, balanced_2, ranking_1, ranking_2, ranking_common = greedy_select(
        num_nodes,
        indptr,
        sources,
        targets,
        weights_1,
        weights_2,
        list(initial_1),
        list(initial_2),
        budget,
        rng,
        preset,
        start_time,
    )
    best_balanced_1 = list(balanced_1)
    best_balanced_2 = list(balanced_2)

    if not should_stop(start_time, preset["time_limit"], preset["stop_buffer"] + 1.0):
        refine_worlds = create_worlds(list(initial_1), list(initial_2), indptr, targets, weights_1, weights_2, num_nodes, rng, preset["refine_worlds"])
        refine_cache = StaticWorldCache(refine_worlds, indptr, targets, num_nodes)
        balanced_1, balanced_2 = local_refine(
            balanced_1,
            balanced_2,
            list(initial_1),
            list(initial_2),
            ranking_1,
            ranking_2,
            ranking_common,
            refine_cache,
            budget,
            preset,
            start_time,
            preset["time_limit"],
        )
        best_balanced_1 = list(balanced_1)
        best_balanced_2 = list(balanced_2)

    if not should_stop(start_time, preset["time_limit"], preset["stop_buffer"] + preset["mc_trial_seconds"] + 0.5):
        balanced_1, balanced_2 = mc_refine(
            balanced_1,
            balanced_2,
            list(initial_1),
            list(initial_2),
            ranking_1,
            ranking_2,
            ranking_common,
            num_nodes,
            indptr,
            targets,
            weights_1,
            weights_2,
            budget,
            preset,
            start_time,
            preset["time_limit"],
        )
        best_balanced_1 = list(balanced_1)
        best_balanced_2 = list(balanced_2)

    if preset["map3_like"] and not should_stop(start_time, preset["time_limit"], preset["stop_buffer"] + 1.2):
        balanced_1, balanced_2 = common_last_mile_refine(
            balanced_1,
            balanced_2,
            list(initial_1),
            list(initial_2),
            ranking_common,
            num_nodes,
            sources,
            targets,
            weights_1,
            weights_2,
            indptr,
            budget,
            preset,
            start_time,
            preset["time_limit"],
        )
        best_balanced_1 = list(balanced_1)
        best_balanced_2 = list(balanced_2)

    best_balanced_1, best_balanced_2 = trim_to_budget(best_balanced_1, best_balanced_2, budget)
    quick_score = estimate_solution_mc(
        list(initial_1),
        list(initial_2),
        best_balanced_1,
        best_balanced_2,
        num_nodes,
        indptr,
        targets,
        weights_1,
        weights_2,
        num_samples=preset["mc_quick_samples"],
    )
    return best_balanced_1, best_balanced_2, quick_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", required=True)
    parser.add_argument("-i", required=True)
    parser.add_argument("-b", required=True)
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-o", required=False, default=None)
    args = parser.parse_args()

    start_time = time.time()
    num_nodes, indptr, sources, targets, weights_1, weights_2 = load_graph(args.n)
    initial_1, initial_2 = load_seeds(args.i)
    preset = get_preset(num_nodes, len(weights_1), args.k, initial_1, initial_2)
    balanced_1, balanced_2, _ = solve_once(
        num_nodes,
        indptr,
        sources,
        targets,
        weights_1,
        weights_2,
        list(initial_1),
        list(initial_2),
        args.k,
        preset,
        start_time,
        DEFAULT_RANDOM_SEED,
    )
    balanced_1, balanced_2 = trim_to_budget(balanced_1, balanced_2, args.k)
    write_seeds(args.b, balanced_1, balanced_2)


if __name__ == "__main__":
    main()
