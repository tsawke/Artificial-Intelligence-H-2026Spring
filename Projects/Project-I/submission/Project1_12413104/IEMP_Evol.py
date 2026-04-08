#!/usr/bin/env python3
"""
IEMP_Evol.py - Evolutionary Algorithm for IEM.
Uses greedy warm-start, ternary encoding, tournament selection,
two-point crossover, hybrid mutation, and a strictly time-bounded
simulated annealing post-refinement over the current best solution.
Usage: python IEMP_Evol.py -n <network> -i <initial_seed> -b <balanced_seed> -k <budget>
"""

import argparse
import time
import numpy as np


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def time_left(start_time, time_limit):
    return time_limit - (time.time() - start_time)


def should_stop(start_time, time_limit, stop_buffer):
    return time_left(start_time, time_limit) <= stop_buffer


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_graph(filepath):
    with open(filepath, 'r') as f:
        n, m = map(int, f.readline().split())
        src = np.empty(m, dtype=np.int32)
        dst = np.empty(m, dtype=np.int32)
        w1 = np.empty(m, dtype=np.float64)
        w2 = np.empty(m, dtype=np.float64)
        for idx in range(m):
            parts = f.readline().split()
            src[idx] = int(parts[0])
            dst[idx] = int(parts[1])
            w1[idx] = float(parts[2])
            w2[idx] = float(parts[3])
    order = np.argsort(src, kind='mergesort')
    dst_s = dst[order]
    w1_s = w1[order]
    w2_s = w2[order]
    counts = np.bincount(src[order], minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return n, m, indptr, dst_s, w1_s, w2_s


def load_seeds(filepath):
    with open(filepath, 'r') as f:
        k1, k2 = map(int, f.readline().split())
        c1 = [int(f.readline().strip()) for _ in range(k1)]
        c2 = [int(f.readline().strip()) for _ in range(k2)]
    return c1, c2


def write_seeds(filepath, s1, s2):
    with open(filepath, 'w') as f:
        f.write(f"{len(s1)} {len(s2)}\n")
        for v in s1:
            f.write(f"{v}\n")
        for v in s2:
            f.write(f"{v}\n")


# ---------------------------------------------------------------------------
# BFS helpers
# ---------------------------------------------------------------------------

def bfs_active(seeds, adj_live):
    active = set(seeds)
    frontier = list(seeds)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj_live.get(u, ()):
                if v not in active:
                    active.add(v)
                    nxt.append(v)
        frontier = nxt
    return active


def compute_exposed(active_set, indptr, targets):
    exposed = set(active_set)
    for u in active_set:
        si = int(indptr[u])
        ei = int(indptr[u + 1])
        for j in range(si, ei):
            exposed.add(int(targets[j]))
    return exposed


def bfs_incremental(v, adj_live, old_active):
    if v in old_active:
        return set()
    reached = {v}
    frontier = [v]
    while frontier:
        nxt = []
        for u in frontier:
            for w in adj_live.get(u, ()):
                if w not in reached and w not in old_active:
                    reached.add(w)
                    nxt.append(w)
        frontier = nxt
    return reached


def build_live_adj(indptr, targets, live_mask, n):
    adj = {}
    for u in range(n):
        si = int(indptr[u])
        ei = int(indptr[u + 1])
        nbrs = []
        for j in range(si, ei):
            if live_mask[j]:
                nbrs.append(int(targets[j]))
        if nbrs:
            adj[u] = nbrs
    return adj


def solution_cost(seeds1, seeds2):
    return len(seeds1) + len(seeds2)


def chrom_cost(chrom):
    return int(np.count_nonzero(chrom == 1) + np.count_nonzero(chrom == 2) + 2 * np.count_nonzero(chrom == 3))


def add_seed_once(seed_list, seed_set, node):
    if node not in seed_set:
        seed_list.append(node)
        seed_set.add(node)


# ---------------------------------------------------------------------------
# World cache for fitness evaluation
# ---------------------------------------------------------------------------

class WorldCache:
    def __init__(self, n, indptr, targets, w1, w2, n_worlds, rng):
        self.n = n
        self.indptr = indptr
        self.targets = targets
        m = len(w1)
        self.worlds = []
        for _ in range(n_worlds):
            live1 = rng.random(m) < w1
            live2 = rng.random(m) < w2
            adj1 = build_live_adj(indptr, targets, live1, n)
            adj2 = build_live_adj(indptr, targets, live2, n)
            self.worlds.append((adj1, adj2))
        self.n_worlds = n_worlds

    def evaluate(self, seeds1, seeds2):
        n = self.n
        indptr = self.indptr
        targets = self.targets
        total = 0.0
        for adj1, adj2 in self.worlds:
            act1 = bfs_active(seeds1, adj1)
            exp1 = compute_exposed(act1, indptr, targets)
            act2 = bfs_active(seeds2, adj2)
            exp2 = compute_exposed(act2, indptr, targets)
            total += n - len(exp1.symmetric_difference(exp2))
        return total / self.n_worlds


# ---------------------------------------------------------------------------
# Greedy warm-start
# ---------------------------------------------------------------------------

def greedy_warmstart(n, indptr, targets, w1, w2, init1, init2, budget,
                     candidate_pool, rng, n_worlds, time_limit_greedy, t_start, stop_buffer):
    m = len(w1)

    worlds = []
    for _ in range(n_worlds):
        live1 = rng.random(m) < w1
        live2 = rng.random(m) < w2
        adj1 = build_live_adj(indptr, targets, live1, n)
        adj2 = build_live_adj(indptr, targets, live2, n)
        act1 = bfs_active(list(init1), adj1)
        exp1 = compute_exposed(act1, indptr, targets)
        act2 = bfs_active(list(init2), adj2)
        exp2 = compute_exposed(act2, indptr, targets)
        worlds.append([adj1, adj2, act1, exp1, act2, exp2,
                       n - len(exp1.symmetric_difference(exp2))])

    init_set1 = set(init1)
    init_set2 = set(init2)
    sel_set1 = set()
    sel_set2 = set()
    bal1, bal2 = [], []

    while solution_cost(bal1, bal2) < budget:
        if should_stop(t_start, time_limit_greedy, stop_buffer):
            break
        best_gain = -1e18
        best_v = -1
        best_camp = -1

        for v in candidate_pool:
            if should_stop(t_start, time_limit_greedy, stop_buffer):
                break
            for camp in [1, 2, 3]:
                if should_stop(t_start, time_limit_greedy, stop_buffer):
                    break
                if camp == 1 and (v in init_set1 or v in sel_set1):
                    continue
                if camp == 2 and (v in init_set2 or v in sel_set2):
                    continue
                if camp == 3 and (v in init_set1 or v in init_set2 or v in sel_set1 or v in sel_set2):
                    continue
                if solution_cost(bal1, bal2) + (2 if camp == 3 else 1) > budget:
                    continue
                gain = 0.0
                for ws in worlds:
                    adj1, adj2, act1, exp1, act2, exp2, base_obj = ws
                    if camp in (1, 3):
                        nr = bfs_incremental(v, adj1, act1)
                        ne1 = set(exp1)
                        if nr:
                            ne1.update(nr)
                            for u in nr:
                                si, ei = int(indptr[u]), int(indptr[u + 1])
                                for j in range(si, ei):
                                    ne1.add(int(targets[j]))
                    else:
                        ne1 = exp1
                    if camp in (2, 3):
                        nr = bfs_incremental(v, adj2, act2)
                        ne2 = set(exp2)
                        if nr:
                            ne2.update(nr)
                            for u in nr:
                                si, ei = int(indptr[u]), int(indptr[u + 1])
                                for j in range(si, ei):
                                    ne2.add(int(targets[j]))
                    else:
                        ne2 = exp2
                    new_obj = n - len(ne1.symmetric_difference(ne2))
                    gain += new_obj - base_obj
                gain = gain / len(worlds)
                score = gain / (2.0 if camp == 3 else 1.0)
                if score > best_gain:
                    best_gain = score
                    best_v = v
                    best_camp = camp

        if best_v < 0:
            break
        if best_camp in (1, 3):
            add_seed_once(bal1, sel_set1, best_v)
        if best_camp in (2, 3):
            add_seed_once(bal2, sel_set2, best_v)

        for ws in worlds:
            adj1, adj2, act1, exp1, act2, exp2, _ = ws
            if best_camp in (1, 3):
                nr = bfs_incremental(best_v, adj1, act1)
                if nr:
                    act1.update(nr)
                    exp1.update(nr)
                    for u in nr:
                        si, ei = int(indptr[u]), int(indptr[u + 1])
                        for j in range(si, ei):
                            exp1.add(int(targets[j]))
            if best_camp in (2, 3):
                nr = bfs_incremental(best_v, adj2, act2)
                if nr:
                    act2.update(nr)
                    exp2.update(nr)
                    for u in nr:
                        si, ei = int(indptr[u]), int(indptr[u + 1])
                        for j in range(si, ei):
                            exp2.add(int(targets[j]))
            ws[6] = n - len(exp1.symmetric_difference(exp2))

    return bal1, bal2


# ---------------------------------------------------------------------------
# EA encoding / operators
# ---------------------------------------------------------------------------

def decode(chrom, candidate_pool):
    s1, s2 = [], []
    for i, g in enumerate(chrom):
        if g == 1:
            s1.append(candidate_pool[i])
        elif g == 2:
            s2.append(candidate_pool[i])
        elif g == 3:
            s1.append(candidate_pool[i])
            s2.append(candidate_pool[i])
    return s1, s2


def repair(chrom, budget, rng, score1, score2, score_common):
    while chrom_cost(chrom) > budget:
        selected = np.nonzero(chrom)[0]
        if len(selected) == 0:
            break
        weakest_idx = None
        weakest_value = None
        for idx in selected:
            state = int(chrom[idx])
            if state == 1:
                value = score1[idx]
            elif state == 2:
                value = score2[idx]
            else:
                value = 0.55 * score_common[idx] + 0.20 * max(score1[idx], score2[idx])
            key = (value / (2.0 if state == 3 else 1.0), value, -int(idx))
            if weakest_value is None or key < weakest_value:
                weakest_value = key
                weakest_idx = int(idx)
        if weakest_idx is None:
            break
        if chrom[weakest_idx] == 3 and chrom_cost(chrom) - budget == 1:
            chrom[weakest_idx] = 1 if score1[weakest_idx] >= score2[weakest_idx] else 2
        elif chrom[weakest_idx] == 3 and rng.random() < 0.55:
            chrom[weakest_idx] = 1 if score1[weakest_idx] >= score2[weakest_idx] else 2
        else:
            chrom[weakest_idx] = 0
    return chrom


def encode_solution(bal1, bal2, candidate_pool):
    pool_idx = {v: i for i, v in enumerate(candidate_pool)}
    chrom = np.zeros(len(candidate_pool), dtype=np.int8)
    for v in bal1:
        if v in pool_idx:
            chrom[pool_idx[v]] = 1
    for v in bal2:
        if v in pool_idx:
            idx = pool_idx[v]
            chrom[idx] = 3 if chrom[idx] == 1 else 2
    return chrom


def solution_key(seeds1, seeds2):
    return tuple(sorted(seeds1)), tuple(sorted(seeds2))


def evaluate_solution_cached(seeds1, seeds2, init1, init2, world_cache, score_cache):
    key = solution_key(seeds1, seeds2)
    if key not in score_cache:
        score_cache[key] = world_cache.evaluate(list(init1) + list(key[0]), list(init2) + list(key[1]))
    return score_cache[key]


def tournament_select(population, fitnesses, rng, k=3):
    indices = rng.choice(len(population), size=min(k, len(population)),
                         replace=False)
    return indices[np.argmax(fitnesses[indices])]


def crossover_two_point(p1, p2, rng):
    length = len(p1)
    if length < 2:
        return p1.copy(), p2.copy()
    pts = sorted(rng.choice(length, size=2, replace=False))
    c1, c2 = p1.copy(), p2.copy()
    c1[pts[0]:pts[1]] = p2[pts[0]:pts[1]]
    c2[pts[0]:pts[1]] = p1[pts[0]:pts[1]]
    return c1, c2


def mutate(chrom, rate, rng):
    for i in range(len(chrom)):
        if rng.random() < rate:
            chrom[i] = rng.integers(0, 4)
    return chrom


def mutate_swap(chrom, rng):
    selected = np.nonzero(chrom)[0]
    if len(selected) == 0:
        return chrom
    idx = rng.choice(selected)
    options = [0, 1, 2, 3]
    options.remove(chrom[idx])
    chrom[idx] = rng.choice(options)
    return chrom


def mutate_add(chrom, budget, rng):
    if chrom_cost(chrom) >= budget:
        return chrom
    unselected = np.where(chrom == 0)[0]
    if len(unselected) == 0:
        return chrom
    idx = rng.choice(unselected)
    remaining = budget - chrom_cost(chrom)
    if remaining >= 2 and rng.random() < 0.25:
        chrom[idx] = 3
    else:
        chrom[idx] = 1 if rng.random() < 0.5 else 2
    return chrom


def sample_sa_neighbor(current_1, current_2, candidate_pool, budget, rng, window_size):
    set_1 = set(current_1)
    set_2 = set(current_2)
    selected_union = set_1 | set_2
    window_add_1 = []
    window_add_2 = []
    window_common = []
    for node in candidate_pool:
        if node not in set_1:
            window_add_1.append(node)
        if node not in set_2:
            window_add_2.append(node)
        if node not in selected_union:
            window_common.append(node)
        if (
            len(window_add_1) >= window_size
            and len(window_add_2) >= window_size
            and len(window_common) >= max(2, window_size // 2)
        ):
            break

    only_1 = [node for node in current_1 if node not in set_2]
    only_2 = [node for node in current_2 if node not in set_1]
    common = [node for node in current_1 if node in set_2]
    current_cost = solution_cost(current_1, current_2)

    operations = []
    if current_1 and window_add_1:
        operations.append("swap1")
    if current_2 and window_add_2:
        operations.append("swap2")
    if common and window_common:
        operations.append("swap_both")
    if only_1:
        operations.append("move12")
    if only_2:
        operations.append("move21")
    if current_cost < budget and window_add_1:
        operations.append("add1")
    if current_cost < budget and window_add_2:
        operations.append("add2")
    if current_cost <= budget - 2 and window_common:
        operations.append("add_both")
    if current_cost < budget and only_1:
        operations.append("commonize1")
    if current_cost < budget and only_2:
        operations.append("commonize2")
    if common:
        operations.append("decommonize1")
        operations.append("decommonize2")
    if only_1 and only_2 and window_common:
        operations.append("replace_pair_with_both")

    if not operations:
        return list(current_1), list(current_2)

    operation = rng.choice(operations)
    trial_1 = list(current_1)
    trial_2 = list(current_2)

    if operation == "swap1":
        remove_idx = int(rng.integers(len(trial_1)))
        replacement = window_add_1[int(rng.integers(len(window_add_1)))]
        trial_1[remove_idx] = replacement
    elif operation == "swap2":
        remove_idx = int(rng.integers(len(trial_2)))
        replacement = window_add_2[int(rng.integers(len(window_add_2)))]
        trial_2[remove_idx] = replacement
    elif operation == "swap_both":
        replacement = window_common[int(rng.integers(len(window_common)))]
        removed = common[int(rng.integers(len(common)))]
        trial_1 = [node for node in trial_1 if node != removed] + [replacement]
        trial_2 = [node for node in trial_2 if node != removed] + [replacement]
    elif operation == "move12":
        moving = only_1[int(rng.integers(len(only_1)))]
        trial_1.remove(moving)
        trial_2.append(moving)
    elif operation == "move21":
        moving = only_2[int(rng.integers(len(only_2)))]
        trial_2.remove(moving)
        trial_1.append(moving)
    elif operation == "add1":
        trial_1.append(window_add_1[int(rng.integers(len(window_add_1)))])
    elif operation == "add2":
        trial_2.append(window_add_2[int(rng.integers(len(window_add_2)))])
    elif operation == "add_both":
        added = window_common[int(rng.integers(len(window_common)))]
        trial_1.append(added)
        trial_2.append(added)
    elif operation == "commonize1":
        trial_2.append(only_1[int(rng.integers(len(only_1)))])
    elif operation == "commonize2":
        trial_1.append(only_2[int(rng.integers(len(only_2)))])
    elif operation == "decommonize1":
        removed = common[int(rng.integers(len(common)))]
        trial_1.remove(removed)
    elif operation == "decommonize2":
        removed = common[int(rng.integers(len(common)))]
        trial_2.remove(removed)
    elif operation == "replace_pair_with_both":
        replacement = window_common[int(rng.integers(len(window_common)))]
        rem1 = only_1[int(rng.integers(len(only_1)))]
        rem2 = only_2[int(rng.integers(len(only_2)))]
        trial_1 = [node for node in trial_1 if node != rem1] + [replacement]
        trial_2 = [node for node in trial_2 if node != rem2] + [replacement]

    return sorted(set(trial_1)), sorted(set(trial_2))


def get_ea_preset(n, m, budget, pool_size):
    exact = {
        (475, 13289): {
            "time_limit": 360.0,
            "stop_buffer": 1.5,
            "warm_worlds": 96,
            "population_size": 20,
            "fast_worlds": 56,
            "accurate_worlds": 92,
            "accurate_topk": 4,
            "mutation_rate": 0.06,
            "elite_size": 2,
            "stale_limit": 5,
            "min_generations": 5,
            "sa_reserved_time": 10.0,
            "sa_fast_worlds": 44,
            "sa_accurate_worlds": 76,
            "sa_window": 64,
            "sa_batch_size": 10,
            "sa_accurate_topk": 3,
            "sa_max_steps": 180,
            "sa_min_seconds": 2.0,
            "sa_max_seconds": 12.0,
            "sa_init_temp_ratio": 0.012,
            "sa_min_temp": 0.6,
        },
        (13984, 17319): {
            "time_limit": 740.0,
            "stop_buffer": 3.0,
            "warm_worlds": 52,
            "population_size": clamp(int(12 + pool_size / 50), 14, 24),
            "fast_worlds": 28,
            "accurate_worlds": 44,
            "accurate_topk": 4,
            "mutation_rate": 0.05,
            "elite_size": 2,
            "stale_limit": 6,
            "min_generations": 6,
            "sa_reserved_time": 20.0,
            "sa_fast_worlds": 28,
            "sa_accurate_worlds": 42,
            "sa_window": 128,
            "sa_batch_size": 12,
            "sa_accurate_topk": 4,
            "sa_max_steps": 260,
            "sa_min_seconds": 4.0,
            "sa_max_seconds": 24.0,
            "sa_init_temp_ratio": 0.008,
            "sa_min_temp": 0.9,
        },
        (3454, 32140): {
            "time_limit": 1200.0,
            "stop_buffer": 3.0,
            "warm_worlds": 64,
            "population_size": clamp(int(12 + pool_size / 48), 14, 22),
            "fast_worlds": 34,
            "accurate_worlds": 56,
            "accurate_topk": 4,
            "mutation_rate": 0.05,
            "elite_size": 2,
            "stale_limit": 6,
            "min_generations": 6,
            "sa_reserved_time": 18.0,
            "sa_fast_worlds": 32,
            "sa_accurate_worlds": 52,
            "sa_window": 112,
            "sa_batch_size": 12,
            "sa_accurate_topk": 4,
            "sa_max_steps": 240,
            "sa_min_seconds": 4.0,
            "sa_max_seconds": 20.0,
            "sa_init_temp_ratio": 0.008,
            "sa_min_temp": 0.9,
        },
    }
    if (n, m) in exact:
        return exact[(n, m)]

    avg_out_degree = m / max(n, 1)
    if n < 1000:
        return {
            "time_limit": 360.0,
            "stop_buffer": 1.5,
            "warm_worlds": 84,
            "population_size": 18,
            "fast_worlds": 48,
            "accurate_worlds": 84,
            "accurate_topk": 4,
            "mutation_rate": 0.05,
            "elite_size": 2,
            "stale_limit": 5,
            "min_generations": 5,
            "sa_reserved_time": 8.0,
            "sa_fast_worlds": 40,
            "sa_accurate_worlds": 68,
            "sa_window": 56,
            "sa_batch_size": 10,
            "sa_accurate_topk": 3,
            "sa_max_steps": 160,
            "sa_min_seconds": 2.0,
            "sa_max_seconds": 10.0,
            "sa_init_temp_ratio": 0.012,
            "sa_min_temp": 0.6,
        }

    if n < 20000:
        return {
            "time_limit": float(clamp(int(160 + 0.0012 * m + 3 * budget), 180, 740)),
            "stop_buffer": 2.5,
            "warm_worlds": clamp(int(62 - 0.6 * min(avg_out_degree, 20.0)), 32, 56),
            "population_size": clamp(int(12 + pool_size / 52), 14, 22),
            "fast_worlds": clamp(int(40 - 0.3 * min(avg_out_degree, 20.0)), 22, 36),
            "accurate_worlds": clamp(int(62 - 0.4 * min(avg_out_degree, 20.0)), 34, 54),
            "accurate_topk": 4,
            "mutation_rate": 0.045,
            "elite_size": 2,
            "stale_limit": 6,
            "min_generations": 6,
            "sa_reserved_time": 16.0,
            "sa_fast_worlds": clamp(int(34 - 0.2 * min(avg_out_degree, 20.0)), 20, 30),
            "sa_accurate_worlds": clamp(int(52 - 0.25 * min(avg_out_degree, 20.0)), 28, 44),
            "sa_window": clamp(int(7 * budget + 52), 84, 160),
            "sa_batch_size": 12,
            "sa_accurate_topk": 4,
            "sa_max_steps": clamp(int(18 * budget), 180, 280),
            "sa_min_seconds": 4.0,
            "sa_max_seconds": 20.0,
            "sa_init_temp_ratio": 0.008,
            "sa_min_temp": 0.9,
        }

    return {
        "time_limit": float(clamp(int(190 + 0.001 * m + 4 * budget), 220, 1200)),
        "stop_buffer": 3.0,
        "warm_worlds": clamp(int(48 - 0.35 * min(avg_out_degree, 24.0)), 24, 40),
        "population_size": clamp(int(12 + pool_size / 60), 12, 20),
        "fast_worlds": clamp(int(30 - 0.2 * min(avg_out_degree, 24.0)), 18, 28),
        "accurate_worlds": clamp(int(48 - 0.25 * min(avg_out_degree, 24.0)), 24, 42),
        "accurate_topk": 4,
        "mutation_rate": 0.04,
        "elite_size": 2,
        "stale_limit": 6,
        "min_generations": 6,
        "sa_reserved_time": 16.0,
        "sa_fast_worlds": clamp(int(26 - 0.15 * min(avg_out_degree, 24.0)), 16, 24),
        "sa_accurate_worlds": clamp(int(42 - 0.20 * min(avg_out_degree, 24.0)), 22, 36),
        "sa_window": clamp(int(6 * budget + 40), 72, 132),
        "sa_batch_size": 10,
        "sa_accurate_topk": 4,
        "sa_max_steps": clamp(int(16 * budget), 160, 240),
        "sa_min_seconds": 3.5,
        "sa_max_seconds": 18.0,
        "sa_init_temp_ratio": 0.007,
        "sa_min_temp": 1.0,
    }


# ---------------------------------------------------------------------------
# EA main loop
# ---------------------------------------------------------------------------

def run_ea(n, m, indptr, targets, w1, w2, init1, init2, budget,
           candidate_info, ws_bal1, ws_bal2, rng, preset, t_start):

    candidate_pool = candidate_info["pool"]
    score1 = candidate_info["score1"]
    score2 = candidate_info["score2"]
    score_common = candidate_info["score_common"]
    pool_size = len(candidate_pool)
    pop_size = preset["population_size"]
    mut_rate = preset["mutation_rate"]
    elite_size = preset["elite_size"]
    fast_wc = WorldCache(n, indptr, targets, w1, w2, preset["fast_worlds"], rng)
    accurate_wc = WorldCache(n, indptr, targets, w1, w2, preset["accurate_worlds"], rng)
    fitness_cache = {}
    accurate_cache = {}

    def fitness(chrom):
        key = chrom.tobytes()
        if key not in fitness_cache:
            s1, s2 = decode(chrom, candidate_pool)
            fitness_cache[key] = fast_wc.evaluate(list(init1) + s1, list(init2) + s2)
        return fitness_cache[key]

    def accurate_score(chrom):
        key = chrom.tobytes()
        if key not in accurate_cache:
            s1, s2 = decode(chrom, candidate_pool)
            accurate_cache[key] = accurate_wc.evaluate(list(init1) + s1, list(init2) + s2)
        return accurate_cache[key]

    def random_structured_mutation(chrom):
        chrom = mutate(chrom, mut_rate, rng)
        if rng.random() < 0.35:
            chrom = mutate_swap(chrom, rng)
        if rng.random() < 0.30:
            chrom = mutate_add(chrom, budget, rng)
        return repair(chrom, budget, rng, score1, score2, score_common)

    population = []
    fitnesses = np.zeros(pop_size)
    ws_chrom = encode_solution(ws_bal1, ws_bal2, candidate_pool)
    population.append(repair(ws_chrom.copy(), budget, rng, score1, score2, score_common))

    common_chrom = np.zeros(pool_size, dtype=np.int8)
    for idx in range(min(pool_size, budget // 2)):
        common_chrom[idx] = 3
        if chrom_cost(common_chrom) >= budget:
            break
    population.append(repair(common_chrom, budget, rng, score1, score2, score_common))

    for _ in range(max(2, pop_size // 3)):
        population.append(random_structured_mutation(ws_chrom.copy()))

    while len(population) < pop_size:
        chrom = np.zeros(pool_size, dtype=np.int8)
        n_sel = int(rng.integers(1, budget + 1))
        indices = rng.choice(pool_size, size=min(n_sel, pool_size), replace=False)
        for idx in indices:
            chrom[idx] = int(rng.integers(1, 4))
        population.append(repair(chrom, budget, rng, score1, score2, score_common))

    for i in range(pop_size):
        fitnesses[i] = fitness(population[i])

    best_chrom = population[int(np.argmax(fitnesses))].copy()
    best_fit = accurate_score(best_chrom)
    stale = 0
    base_mut = mut_rate
    generations = 0
    ea_stop_buffer = preset["stop_buffer"] + preset["sa_reserved_time"]

    while True:
        if should_stop(t_start, preset["time_limit"], ea_stop_buffer):
            break

        elite_idx = np.argsort(fitnesses)[::-1][:elite_size]
        new_pop = [population[ei].copy() for ei in elite_idx]

        while len(new_pop) < pop_size:
            if should_stop(t_start, preset["time_limit"], ea_stop_buffer):
                break
            p1 = tournament_select(population, fitnesses, rng)
            p2 = tournament_select(population, fitnesses, rng)
            c1, c2 = crossover_two_point(population[p1], population[p2], rng)
            c1 = random_structured_mutation(c1)
            c2 = random_structured_mutation(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        new_fit = np.full(len(new_pop), -1.0e18)
        for i in range(len(new_pop)):
            if should_stop(t_start, preset["time_limit"], ea_stop_buffer):
                new_fit[i] = -1.0e18
            else:
                new_fit[i] = fitness(new_pop[i])

        population = new_pop
        fitnesses = new_fit
        generations += 1

        ranked_idx = np.argsort(fitnesses)[::-1][:preset["accurate_topk"]]
        improved = False
        for idx in ranked_idx:
            score = accurate_score(population[idx])
            if score > best_fit:
                best_fit = score
                best_chrom = population[idx].copy()
                improved = True

        if improved:
            stale = 0
            mut_rate = base_mut
        else:
            stale += 1
            if stale > preset["stale_limit"] and generations >= preset["min_generations"]:
                break
            if stale > max(2, preset["stale_limit"] // 2):
                mut_rate = min(0.20, mut_rate * 1.35)

    return decode(best_chrom, candidate_pool)


def run_sa_post_refinement(n, indptr, targets, w1, w2, init1, init2, budget,
                           candidate_pool, seeds1, seeds2, rng, preset, t_start):
    remaining_time = time_left(t_start, preset["time_limit"]) - preset["stop_buffer"]
    usable_time = min(preset["sa_max_seconds"], remaining_time)
    if usable_time < preset["sa_min_seconds"]:
        return seeds1, seeds2

    sa_start = time.time()
    fast_wc = WorldCache(n, indptr, targets, w1, w2, preset["sa_fast_worlds"], rng)
    accurate_wc = WorldCache(n, indptr, targets, w1, w2, preset["sa_accurate_worlds"], rng)
    fast_cache = {}
    accurate_cache = {}
    current_1 = sorted(set(seeds1))
    current_2 = sorted(set(seeds2))
    current_score = evaluate_solution_cached(current_1, current_2, init1, init2, accurate_wc, accurate_cache)
    best_1 = list(current_1)
    best_2 = list(current_2)
    best_score = current_score
    initial_temperature = max(preset["sa_min_temp"], current_score * preset["sa_init_temp_ratio"])

    for step in range(preset["sa_max_steps"]):
        if should_stop(t_start, preset["time_limit"], preset["stop_buffer"]):
            break
        elapsed = time.time() - sa_start
        if elapsed >= usable_time:
            break

        progress = elapsed / max(usable_time, 1.0e-9)
        temperature = max(
            preset["sa_min_temp"],
            initial_temperature * (1.0 - 0.9 * progress),
        )

        trial_batch = []
        trial_seen = set()
        for _ in range(preset["sa_batch_size"]):
            trial_1, trial_2 = sample_sa_neighbor(
                current_1,
                current_2,
                candidate_pool,
                budget,
                rng,
                preset["sa_window"],
            )
            key = solution_key(trial_1, trial_2)
            if key == solution_key(current_1, current_2) or key in trial_seen:
                continue
            trial_seen.add(key)
            trial_batch.append((trial_1, trial_2))
        if not trial_batch:
            continue

        fast_scores = []
        for trial_1, trial_2 in trial_batch:
            fast_score = evaluate_solution_cached(trial_1, trial_2, init1, init2, fast_wc, fast_cache)
            fast_scores.append((fast_score, trial_1, trial_2))
        fast_scores.sort(key=lambda item: item[0], reverse=True)

        finalists = fast_scores[:preset["sa_accurate_topk"]]
        trial_score = None
        chosen_trial = None
        for _, trial_1, trial_2 in finalists:
            accurate = evaluate_solution_cached(trial_1, trial_2, init1, init2, accurate_wc, accurate_cache)
            if trial_score is None or accurate > trial_score:
                trial_score = accurate
                chosen_trial = (trial_1, trial_2)
        if chosen_trial is None:
            continue

        trial_1, trial_2 = chosen_trial
        delta = trial_score - current_score
        if delta >= 0.0:
            accept = True
        else:
            accept = rng.random() < np.exp(delta / max(temperature, 1.0e-9))

        if not accept:
            continue

        current_1 = trial_1
        current_2 = trial_2
        current_score = trial_score

        if current_score > best_score:
            best_1 = list(current_1)
            best_2 = list(current_2)
            best_score = current_score

    return best_1, best_2


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

def build_candidate_pool(n, indptr, targets, w1, w2, init1, init2, max_pool):
    all_seeds = set(init1) | set(init2)
    out1 = np.zeros(n)
    out2 = np.zeros(n)
    for u in range(n):
        si, ei = int(indptr[u]), int(indptr[u + 1])
        if si < ei:
            out1[u] = np.sum(w1[si:ei])
            out2[u] = np.sum(w2[si:ei])
    in1 = np.zeros(n)
    in2 = np.zeros(n)
    m = len(w1)
    for j in range(m):
        v = int(targets[j])
        in1[v] += w1[j]
        in2[v] += w2[j]
    asymmetry = np.abs((out1 + in1) - (out2 + in2))
    score1 = out1 + 0.35 * in1 + 0.20 * asymmetry
    score2 = out2 + 0.35 * in2 + 0.20 * asymmetry
    score_common = np.minimum(score1, score2) + 0.25 * (score1 + score2)
    structural = out1 + out2 + in1 + in2
    rank1 = np.argsort(score1)[::-1]
    rank2 = np.argsort(score2)[::-1]
    rank_common = np.argsort(score_common)[::-1]
    rank_structural = np.argsort(structural)[::-1]
    merged = []
    seen = set()
    for ranking in (rank1, rank2, rank_common, rank_structural):
        for v in ranking:
            node = int(v)
            if node in all_seeds or node in seen:
                continue
            seen.add(node)
            merged.append(node)
            if len(merged) >= max_pool:
                break
        if len(merged) >= max_pool:
            break
    pool_index = {node: idx for idx, node in enumerate(merged)}
    pool_score1 = np.array([score1[node] for node in merged], dtype=np.float64)
    pool_score2 = np.array([score2[node] for node in merged], dtype=np.float64)
    pool_score_common = np.array([score_common[node] for node in merged], dtype=np.float64)
    return {
        "pool": merged,
        "index": pool_index,
        "score1": pool_score1,
        "score2": pool_score2,
        "score_common": pool_score_common,
        "rank1": [int(node) for node in rank1 if int(node) in pool_index],
        "rank2": [int(node) for node in rank2 if int(node) in pool_index],
        "rank_common": [int(node) for node in rank_common if int(node) in pool_index],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', required=True)
    parser.add_argument('-i', required=True)
    parser.add_argument('-b', required=True)
    parser.add_argument('-k', type=int, required=True)
    parser.add_argument('-o', required=False, default=None)
    args = parser.parse_args()

    t_start = time.time()
    n, m, indptr, targets, w1, w2 = load_graph(args.n)
    init1, init2 = load_seeds(args.i)
    budget = args.k

    rng = np.random.default_rng(54321)

    if n < 1000:
        max_pool = n
    elif n < 20000:
        max_pool = clamp(int(16 * budget + 1.5 * np.sqrt(n)), 140, 520)
    else:
        max_pool = clamp(int(18 * budget + 1.2 * np.sqrt(n)), 160, 360)

    candidate_info = build_candidate_pool(n, indptr, targets, w1, w2, init1, init2, max_pool)
    candidate_pool = candidate_info["pool"]
    preset = get_ea_preset(n, m, budget, len(candidate_pool))

    greedy_time = preset["time_limit"] * 0.22
    ws_bal1, ws_bal2 = greedy_warmstart(
        n, indptr, targets, w1, w2, init1, init2, budget,
        candidate_pool, rng, preset["warm_worlds"], greedy_time, t_start, preset["stop_buffer"])

    s1, s2 = ws_bal1, ws_bal2
    if not should_stop(t_start, preset["time_limit"], preset["stop_buffer"] + preset["sa_reserved_time"]):
        s1, s2 = run_ea(n, m, indptr, targets, w1, w2, init1, init2, budget,
                        candidate_info, ws_bal1, ws_bal2, rng, preset, t_start)

    if not should_stop(t_start, preset["time_limit"], preset["stop_buffer"]):
        s1, s2 = run_sa_post_refinement(
            n,
            indptr,
            targets,
            w1,
            w2,
            init1,
            init2,
            budget,
            candidate_pool,
            s1,
            s2,
            rng,
            preset,
            t_start,
        )

    s1 = sorted(set(s1))
    s2 = sorted(set(s2))
    while solution_cost(s1, s2) > budget:
        if len(s1) >= len(s2) and s1:
            s1.pop()
        elif s2:
            s2.pop()
        else:
            break
    write_seeds(args.b, s1, s2)


if __name__ == '__main__':
    main()
