from __future__ import annotations

import math
import os
from dataclasses import dataclass

from Evaluator import (
    RandomContext,
    SeedSets,
    Timer,
    load_graph,
    load_seed_sets,
    parse_common_args,
    validate_seed_sets,
    write_seed_sets,
)
from IEMP_Heur import (
    SplitProfile,
    build_campaign_candidate_pool,
    build_guard_baseline_solution,
    build_common_candidate_pool,
    build_heuristic_solution,
    build_precomputed_data,
    build_ranked_candidates,
    build_submission_fallback_solution,
    build_search_state_from_solution,
    build_timeout_baseline_solution,
    compute_campaign_scores,
    construct_solution_for_profile,
    estimate_solution_shared_worlds,
)


SMALL_GRAPH_THRESHOLD = 1024
MAX_ITERS_SMALL = 90
MAX_ITERS_LARGE = 70
FAST_EVAL_WORLDS_SMALL = 12
FAST_EVAL_WORLDS_LARGE = 4
ACCURATE_EVAL_WORLDS_SMALL = 32
ACCURATE_EVAL_WORLDS_LARGE = 8
TEMP_SCALE_SMALL = 0.02
TEMP_SCALE_LARGE = 0.005
COOLING_SMALL = 0.97
COOLING_LARGE = 0.985
CANDIDATE_LIMIT_SMALL = 96
CANDIDATE_LIMIT_LARGE = 128
NEIGHBOR_BATCH_SMALL = 10
NEIGHBOR_BATCH_LARGE = 8
ACCURATE_KEEP_SMALL = 3
ACCURATE_KEEP_LARGE = 2
POOL_FRONTIER_SMALL = 12
POOL_FRONTIER_LARGE = 10
NEIGHBOR_MIN_SECONDS = 0.25
LAST_MILE_MIN_SECONDS_SMALL = 0.45
LAST_MILE_MIN_SECONDS_LARGE = 2.4
LAST_MILE_STEPS_SMALL = 3
LAST_MILE_STEPS_LARGE = 2
LAST_MILE_FAST_KEEP_SMALL = 4
LAST_MILE_FAST_KEEP_LARGE = 3
LAST_MILE_ACCURATE_KEEP_SMALL = 2
LAST_MILE_ACCURATE_KEEP_LARGE = 1
WARM_START_LIMIT_SMALL = 30.0
WARM_START_LIMIT_LARGE = 120.0
LOCAL_RESERVE_SMALL = 6.0
LOCAL_RESERVE_LARGE = 30.0
PRECOMPUTE_MIN_SECONDS_SMALL = 2.0
PRECOMPUTE_MIN_SECONDS_LARGE = 10.0
ACCURATE_EVAL_MIN_SECONDS_SMALL = 0.3
ACCURATE_EVAL_MIN_SECONDS_LARGE = 0.9
START_COUNT_SMALL = 1
START_COUNT_LARGE = 1
STALL_LIMIT_SMALL = 6
STALL_LIMIT_LARGE = 5
PERTURB_STEPS_SMALL = 4
PERTURB_STEPS_LARGE = 3
REHEAT_MULTIPLIER_SMALL = 1.8
REHEAT_MULTIPLIER_LARGE = 1.6
START_GENERATION_MIN_SECONDS_SMALL = 4.0
START_GENERATION_MIN_SECONDS_LARGE = 14.0
PROFILE_START_MIN_SECONDS_SMALL = 1.2
PROFILE_START_MIN_SECONDS_LARGE = 4.0
EVOL_GUARD_RESERVE_SCALE = 1.25


@dataclass(slots=True)
class SparseSolution:
    campaign1: set[int]
    campaign2: set[int]

    @property
    def total_size(self) -> int:
        return len(self.campaign1) + len(self.campaign2)

    def copy(self) -> SparseSolution:
        return SparseSolution(campaign1=set(self.campaign1), campaign2=set(self.campaign2))

    def to_seed_sets(self) -> SeedSets:
        return SeedSets(
            campaign1=sorted(self.campaign1),
            campaign2=sorted(self.campaign2),
        )


def sparse_from_seed_sets(seed_sets: SeedSets) -> SparseSolution:
    return SparseSolution(campaign1=set(seed_sets.campaign1), campaign2=set(seed_sets.campaign2))


def sparse_signature(solution: SparseSolution) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(sorted(solution.campaign1)),
        tuple(sorted(solution.campaign2)),
    )


def infer_evol_time_limit_seconds(graph) -> float:
    if graph.num_nodes <= 1000:
        return 420.0
    if graph.num_nodes == 13984 and graph.num_edges == 17319:
        return 860.0
    if graph.num_nodes == 3454 and graph.num_edges == 32140:
        return 1350.0

    average_out_degree = graph.num_edges / max(1, graph.num_nodes)
    if graph.num_nodes >= 10000:
        return 860.0
    if average_out_degree >= 8.0:
        return 1350.0
    return 420.0


def infer_evol_time_reserve_seconds(time_limit_seconds: float) -> float:
    if time_limit_seconds <= 450.0:
        base_reserve = 25.0
    elif time_limit_seconds <= 900.0:
        base_reserve = 45.0
    else:
        base_reserve = 60.0
    return base_reserve * EVOL_GUARD_RESERVE_SCALE


def create_evol_timer(graph) -> Timer:
    override_limit = os.getenv("IEMP_EVOL_TIME_LIMIT_SECONDS")
    override_reserve = os.getenv("IEMP_EVOL_TIME_RESERVE_SECONDS")

    if override_limit is not None:
        time_limit_seconds = float(override_limit)
    else:
        time_limit_seconds = infer_evol_time_limit_seconds(graph)

    if override_reserve is not None:
        reserve_seconds = float(override_reserve)
    else:
        reserve_seconds = infer_evol_time_reserve_seconds(time_limit_seconds)

    return Timer(time_limit_seconds=time_limit_seconds, reserve_seconds=reserve_seconds)


def warm_start_time_limit(graph, timer: Timer) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        hard_cap = WARM_START_LIMIT_SMALL
        reserve = LOCAL_RESERVE_SMALL
    else:
        hard_cap = WARM_START_LIMIT_LARGE
        reserve = LOCAL_RESERVE_LARGE
    return max(0.0, min(hard_cap, timer.remaining() - reserve))


def candidate_limit(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return CANDIDATE_LIMIT_SMALL
    return CANDIDATE_LIMIT_LARGE


def max_iterations(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return MAX_ITERS_SMALL
    return MAX_ITERS_LARGE


def fast_eval_worlds(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return FAST_EVAL_WORLDS_SMALL
    return FAST_EVAL_WORLDS_LARGE


def accurate_eval_worlds(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return ACCURATE_EVAL_WORLDS_SMALL
    return ACCURATE_EVAL_WORLDS_LARGE


def neighbor_batch_size(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return NEIGHBOR_BATCH_SMALL
    return NEIGHBOR_BATCH_LARGE


def accurate_eval_keep(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return ACCURATE_KEEP_SMALL
    return ACCURATE_KEEP_LARGE


def candidate_frontier_size(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return POOL_FRONTIER_SMALL
    return POOL_FRONTIER_LARGE


def precompute_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return PRECOMPUTE_MIN_SECONDS_SMALL
    return PRECOMPUTE_MIN_SECONDS_LARGE


def accurate_eval_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return ACCURATE_EVAL_MIN_SECONDS_SMALL
    return ACCURATE_EVAL_MIN_SECONDS_LARGE


def last_mile_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LAST_MILE_MIN_SECONDS_SMALL
    return LAST_MILE_MIN_SECONDS_LARGE


def last_mile_steps(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LAST_MILE_STEPS_SMALL
    return LAST_MILE_STEPS_LARGE


def last_mile_fast_keep(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LAST_MILE_FAST_KEEP_SMALL
    return LAST_MILE_FAST_KEEP_LARGE


def last_mile_accurate_keep(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LAST_MILE_ACCURATE_KEEP_SMALL
    return LAST_MILE_ACCURATE_KEEP_LARGE


def restart_count(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return START_COUNT_SMALL
    return START_COUNT_LARGE


def iterations_per_start(graph) -> int:
    return max(12, math.ceil(max_iterations(graph) / restart_count(graph)))


def stall_limit(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return STALL_LIMIT_SMALL
    return STALL_LIMIT_LARGE


def perturb_steps(graph) -> int:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return PERTURB_STEPS_SMALL
    return PERTURB_STEPS_LARGE


def reheat_multiplier(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return REHEAT_MULTIPLIER_SMALL
    return REHEAT_MULTIPLIER_LARGE


def start_generation_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return START_GENERATION_MIN_SECONDS_SMALL
    return START_GENERATION_MIN_SECONDS_LARGE


def profile_start_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return PROFILE_START_MIN_SECONDS_SMALL
    return PROFILE_START_MIN_SECONDS_LARGE


def initial_temperature(graph, initial_score: float) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        scale = TEMP_SCALE_SMALL
    else:
        scale = TEMP_SCALE_LARGE
    return max(1.0, abs(initial_score) * scale)


def cooling_rate(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return COOLING_SMALL
    return COOLING_LARGE


def cached_solution_score(
    graph,
    initial_seed_sets: SeedSets,
    solution: SparseSolution,
    num_worlds: int,
    score_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    eval_seed: int,
) -> float:
    signature = sparse_signature(solution)
    cached = score_cache.get(signature)
    if cached is not None:
        return cached

    score = estimate_solution_shared_worlds(
        graph=graph,
        initial_seeds=initial_seed_sets,
        balanced_seed_sets=solution.to_seed_sets(),
        num_worlds=num_worlds,
        random_seed=eval_seed,
    )
    score_cache[signature] = score
    return score


def fast_eval_solution(
    graph,
    initial_seed_sets: SeedSets,
    solution: SparseSolution,
    score_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    eval_seed: int,
) -> float:
    return cached_solution_score(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=solution,
        num_worlds=fast_eval_worlds(graph),
        score_cache=score_cache,
        eval_seed=eval_seed,
    )


def accurate_eval_solution(
    graph,
    initial_seed_sets: SeedSets,
    solution: SparseSolution,
    score_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    eval_seed: int,
) -> float:
    return cached_solution_score(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=solution,
        num_worlds=accurate_eval_worlds(graph),
        score_cache=score_cache,
        eval_seed=eval_seed,
    )


def choose_from_sorted(nodes: list[int], rng, sample_limit: int) -> int | None:
    if not nodes:
        return None
    limit = min(len(nodes), sample_limit)
    return nodes[rng.randrange(limit)]


def available_ranked_nodes(
    ranked_nodes: list[int],
    forbidden_nodes: set[int],
    limit: int,
) -> list[int]:
    candidates: list[int] = []
    for node in ranked_nodes:
        if node in forbidden_nodes:
            continue
        candidates.append(node)
        if len(candidates) >= limit:
            break
    return candidates


def weakest_nodes(nodes: set[int], scores: list[float], limit: int) -> list[int]:
    ranked = sorted(nodes, key=lambda node: (scores[node], node))
    return ranked[: min(limit, len(ranked))]


def sample_ranked_frontier(
    nodes: list[int],
    rng,
    sample_limit: int,
    frontier_limit: int,
) -> list[int]:
    if not nodes:
        return []
    frontier = list(nodes[: min(len(nodes), frontier_limit)])
    if len(frontier) <= sample_limit:
        return frontier

    deterministic_keep = max(1, sample_limit // 2)
    selected = frontier[:deterministic_keep]
    remaining = frontier[deterministic_keep:]
    while len(selected) < sample_limit and remaining:
        idx = rng.randrange(len(remaining))
        selected.append(remaining.pop(idx))
    return selected


def campaign_structural_rank(precomputed, campaign: int, node: int) -> float:
    if campaign == 1:
        return precomputed.coverage_strength1[node] + 0.25 * precomputed.common_strength[node]
    return precomputed.coverage_strength2[node] + 0.25 * precomputed.common_strength[node]


def common_structural_rank(precomputed, node: int) -> float:
    return precomputed.common_strength[node] + 0.35 * precomputed.combined_strength[node]


def exposure_gap_bonus(state, campaign: int, node: int) -> float:
    if campaign == 1:
        return max(0.0, state.approx2[node] - state.approx1[node])
    return max(0.0, state.approx1[node] - state.approx2[node])


def register_sparse_neighbor(
    neighbors: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, SparseSolution]],
    candidate: SparseSolution,
    priority: float,
    budget: int,
    current_signature: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    if candidate.total_size > budget:
        return
    signature = sparse_signature(candidate)
    if signature == current_signature:
        return
    previous = neighbors.get(signature)
    if previous is None or priority > previous[0]:
        neighbors[signature] = (priority, candidate)


def build_structured_neighbors(
    graph,
    initial_seed_sets: SeedSets,
    current: SparseSolution,
    budget: int,
    precomputed,
    rng,
) -> list[SparseSolution]:
    current_signature = sparse_signature(current)
    current_seed_sets = current.to_seed_sets()
    current_state = build_search_state_from_solution(
        graph=graph,
        initial_seeds=initial_seed_sets,
        precomputed=precomputed,
        balanced_seed_sets=current_seed_sets,
    )

    only1 = list(current.campaign1 - current.campaign2)
    only2 = list(current.campaign2 - current.campaign1)
    common = list(current.campaign1 & current.campaign2)
    frontier = candidate_frontier_size(graph)
    sample_width = max(2, min(4, neighbor_batch_size(graph)))

    weak_only1 = sorted(only1, key=lambda node: (campaign_structural_rank(precomputed, 1, node), node))
    weak_only2 = sorted(only2, key=lambda node: (campaign_structural_rank(precomputed, 2, node), node))
    weak_common1 = sorted(common, key=lambda node: (campaign_structural_rank(precomputed, 1, node), node))
    weak_common2 = sorted(common, key=lambda node: (campaign_structural_rank(precomputed, 2, node), node))
    weak_common = sorted(common, key=lambda node: (common_structural_rank(precomputed, node), node))
    commonize_from1 = sorted(only1, key=lambda node: (-(common_structural_rank(precomputed, node)), node))
    commonize_from2 = sorted(only2, key=lambda node: (-(common_structural_rank(precomputed, node)), node))
    move_from1 = sorted(
        only1,
        key=lambda node: (
            -(
                common_structural_rank(precomputed, node)
                + exposure_gap_bonus(current_state, 2, node)
                - 0.35 * campaign_structural_rank(precomputed, 1, node)
            ),
            node,
        ),
    )
    move_from2 = sorted(
        only2,
        key=lambda node: (
            -(
                common_structural_rank(precomputed, node)
                + exposure_gap_bonus(current_state, 1, node)
                - 0.35 * campaign_structural_rank(precomputed, 2, node)
            ),
            node,
        ),
    )

    initial_campaign1 = set(initial_seed_sets.campaign1)
    initial_campaign2 = set(initial_seed_sets.campaign2)
    pool1 = build_campaign_candidate_pool(
        graph=graph,
        campaign=1,
        precomputed=precomputed,
        initial_campaign_nodes=initial_campaign1,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )
    pool2 = build_campaign_candidate_pool(
        graph=graph,
        campaign=2,
        precomputed=precomputed,
        initial_campaign_nodes=initial_campaign2,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )
    common_pool = build_common_candidate_pool(
        graph=graph,
        precomputed=precomputed,
        initial_campaign1=initial_campaign1,
        initial_campaign2=initial_campaign2,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )

    add_nodes1 = sample_ranked_frontier(pool1, rng, sample_width, frontier)
    add_nodes2 = sample_ranked_frontier(pool2, rng, sample_width, frontier)
    common_adds = sample_ranked_frontier(common_pool, rng, max(2, sample_width - 1), frontier)
    delete_only1 = sample_ranked_frontier(weak_only1, rng, sample_width, frontier)
    delete_only2 = sample_ranked_frontier(weak_only2, rng, sample_width, frontier)
    decommon1 = sample_ranked_frontier(weak_common1, rng, max(2, sample_width - 1), frontier)
    decommon2 = sample_ranked_frontier(weak_common2, rng, max(2, sample_width - 1), frontier)
    swap_common_out = sample_ranked_frontier(weak_common, rng, max(2, sample_width - 1), frontier)
    move_nodes1 = sample_ranked_frontier(move_from1, rng, sample_width, frontier)
    move_nodes2 = sample_ranked_frontier(move_from2, rng, sample_width, frontier)
    commonize_nodes1 = sample_ranked_frontier(commonize_from1, rng, sample_width, frontier)
    commonize_nodes2 = sample_ranked_frontier(commonize_from2, rng, sample_width, frontier)

    neighbors: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[float, SparseSolution]] = {}
    budget_headroom = budget - current.total_size

    if budget_headroom >= 1:
        for node in add_nodes1:
            candidate = SparseSolution(campaign1=set(current.campaign1) | {node}, campaign2=set(current.campaign2))
            priority = campaign_structural_rank(precomputed, 1, node) + exposure_gap_bonus(current_state, 1, node)
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)
        for node in add_nodes2:
            candidate = SparseSolution(campaign1=set(current.campaign1), campaign2=set(current.campaign2) | {node})
            priority = campaign_structural_rank(precomputed, 2, node) + exposure_gap_bonus(current_state, 2, node)
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for node in delete_only1:
        candidate = SparseSolution(campaign1=set(current.campaign1) - {node}, campaign2=set(current.campaign2))
        priority = -0.5 * campaign_structural_rank(precomputed, 1, node)
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)
    for node in delete_only2:
        candidate = SparseSolution(campaign1=set(current.campaign1), campaign2=set(current.campaign2) - {node})
        priority = -0.5 * campaign_structural_rank(precomputed, 2, node)
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for removed in delete_only1:
        for added in add_nodes1:
            if added == removed:
                continue
            candidate = SparseSolution(
                campaign1=(set(current.campaign1) - {removed}) | {added},
                campaign2=set(current.campaign2),
            )
            priority = (
                campaign_structural_rank(precomputed, 1, added)
                - 0.65 * campaign_structural_rank(precomputed, 1, removed)
                + exposure_gap_bonus(current_state, 1, added)
            )
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for removed in delete_only2:
        for added in add_nodes2:
            if added == removed:
                continue
            candidate = SparseSolution(
                campaign1=set(current.campaign1),
                campaign2=(set(current.campaign2) - {removed}) | {added},
            )
            priority = (
                campaign_structural_rank(precomputed, 2, added)
                - 0.65 * campaign_structural_rank(precomputed, 2, removed)
                + exposure_gap_bonus(current_state, 2, added)
            )
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for removed in swap_common_out:
        for added in common_adds:
            if added == removed:
                continue
            candidate = SparseSolution(
                campaign1=(set(current.campaign1) - {removed}) | {added},
                campaign2=(set(current.campaign2) - {removed}) | {added},
            )
            priority = common_structural_rank(precomputed, added) - 0.7 * common_structural_rank(precomputed, removed)
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for node in move_nodes1:
        candidate = SparseSolution(
            campaign1=set(current.campaign1) - {node},
            campaign2=set(current.campaign2) | {node},
        )
        priority = (
            common_structural_rank(precomputed, node)
            + exposure_gap_bonus(current_state, 2, node)
            - 0.3 * campaign_structural_rank(precomputed, 1, node)
        )
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for node in move_nodes2:
        candidate = SparseSolution(
            campaign1=set(current.campaign1) | {node},
            campaign2=set(current.campaign2) - {node},
        )
        priority = (
            common_structural_rank(precomputed, node)
            + exposure_gap_bonus(current_state, 1, node)
            - 0.3 * campaign_structural_rank(precomputed, 2, node)
        )
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    if budget_headroom >= 1:
        for node in commonize_nodes1:
            candidate = SparseSolution(campaign1=set(current.campaign1), campaign2=set(current.campaign2) | {node})
            priority = common_structural_rank(precomputed, node) + exposure_gap_bonus(current_state, 2, node)
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)
        for node in commonize_nodes2:
            candidate = SparseSolution(campaign1=set(current.campaign1) | {node}, campaign2=set(current.campaign2))
            priority = common_structural_rank(precomputed, node) + exposure_gap_bonus(current_state, 1, node)
            register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    for node in decommon1:
        candidate = SparseSolution(campaign1=set(current.campaign1) - {node}, campaign2=set(current.campaign2))
        priority = -0.35 * campaign_structural_rank(precomputed, 1, node)
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)
    for node in decommon2:
        candidate = SparseSolution(campaign1=set(current.campaign1), campaign2=set(current.campaign2) - {node})
        priority = -0.35 * campaign_structural_rank(precomputed, 2, node)
        register_sparse_neighbor(neighbors, candidate, priority, budget, current_signature)

    if not neighbors:
        return []

    ranked_neighbors = sorted(
        neighbors.values(),
        key=lambda item: (-item[0], sparse_signature(item[1])),
    )
    shortlist = ranked_neighbors[: max(neighbor_batch_size(graph) * 2, accurate_eval_keep(graph))]
    chosen = shortlist[: max(1, neighbor_batch_size(graph) // 2)]
    remaining = shortlist[len(chosen) :]
    while len(chosen) < min(len(shortlist), neighbor_batch_size(graph)) and remaining:
        idx = rng.randrange(len(remaining))
        chosen.append(remaining.pop(idx))
    return [candidate for _, candidate in chosen]


def propose_basic_neighbor(
    graph,
    current: SparseSolution,
    budget: int,
    ranked1: list[int],
    ranked2: list[int],
    score1: list[float],
    score2: list[float],
    rng,
) -> SparseSolution | None:
    forbidden_union = current.campaign1 | current.campaign2
    pool_limit = candidate_limit(graph)
    add_pool1 = available_ranked_nodes(ranked1, forbidden_union, pool_limit)
    add_pool2 = available_ranked_nodes(ranked2, forbidden_union, pool_limit)

    actions: list[str] = []
    if current.total_size < budget and add_pool1:
        actions.append("add1")
    if current.total_size < budget and add_pool2:
        actions.append("add2")
    if current.campaign1:
        actions.append("del1")
    if current.campaign2:
        actions.append("del2")
    if current.campaign1 and add_pool1:
        actions.append("swap1")
    if current.campaign2 and add_pool2:
        actions.append("swap2")

    if not actions:
        return None

    action = actions[rng.randrange(len(actions))]
    candidate = current.copy()

    if action == "add1":
        node = choose_from_sorted(add_pool1, rng, 8)
        if node is None:
            return None
        candidate.campaign1.add(node)
        return candidate

    if action == "add2":
        node = choose_from_sorted(add_pool2, rng, 8)
        if node is None:
            return None
        candidate.campaign2.add(node)
        return candidate

    if action == "del1":
        removable = weakest_nodes(candidate.campaign1, score1, 6)
        node = choose_from_sorted(removable, rng, 3)
        if node is None:
            return None
        candidate.campaign1.remove(node)
        return candidate

    if action == "del2":
        removable = weakest_nodes(candidate.campaign2, score2, 6)
        node = choose_from_sorted(removable, rng, 3)
        if node is None:
            return None
        candidate.campaign2.remove(node)
        return candidate

    if action == "swap1":
        removable = weakest_nodes(candidate.campaign1, score1, 6)
        remove_node = choose_from_sorted(removable, rng, 3)
        add_node = choose_from_sorted(add_pool1, rng, 8)
        if remove_node is None or add_node is None or add_node == remove_node:
            return None
        candidate.campaign1.remove(remove_node)
        candidate.campaign1.add(add_node)
        return candidate

    removable = weakest_nodes(candidate.campaign2, score2, 6)
    remove_node = choose_from_sorted(removable, rng, 3)
    add_node = choose_from_sorted(add_pool2, rng, 8)
    if remove_node is None or add_node is None or add_node == remove_node:
        return None
    candidate.campaign2.remove(remove_node)
    candidate.campaign2.add(add_node)
    return candidate


def build_basic_neighbor_batch(
    graph,
    current: SparseSolution,
    budget: int,
    ranked1: list[int],
    ranked2: list[int],
    score1: list[float],
    score2: list[float],
    rng,
) -> list[SparseSolution]:
    neighbors: dict[tuple[tuple[int, ...], tuple[int, ...]], SparseSolution] = {}
    attempts = neighbor_batch_size(graph) * 6
    while len(neighbors) < neighbor_batch_size(graph) and attempts > 0:
        attempts -= 1
        candidate = propose_basic_neighbor(
            graph=graph,
            current=current,
            budget=budget,
            ranked1=ranked1,
            ranked2=ranked2,
            score1=score1,
            score2=score2,
            rng=rng,
        )
        if candidate is None:
            continue
        neighbors.setdefault(sparse_signature(candidate), candidate)
    return list(neighbors.values())


def choose_transition_candidate(scored_candidates, rng):
    if len(scored_candidates) == 1:
        return scored_candidates[0]

    total_weight = 0
    for index in range(len(scored_candidates)):
        total_weight += len(scored_candidates) - index

    threshold = rng.randrange(total_weight)
    running = 0
    for index, item in enumerate(scored_candidates):
        running += len(scored_candidates) - index
        if threshold < running:
            return item
    return scored_candidates[0]


def register_start_solution(
    destination: dict[tuple[tuple[int, ...], tuple[int, ...]], SparseSolution],
    candidate,
) -> None:
    sparse_candidate = candidate if isinstance(candidate, SparseSolution) else sparse_from_seed_sets(candidate)
    destination.setdefault(sparse_signature(sparse_candidate), sparse_candidate)


def initial_strength_gap(score1: list[float], score2: list[float], initial_seed_sets: SeedSets) -> int:
    strength1 = sum(score1[node] for node in initial_seed_sets.campaign1)
    strength2 = sum(score2[node] for node in initial_seed_sets.campaign2)
    if strength1 <= strength2:
        return 1
    return 2


def total_gap(own_exposure: list[float], other_exposure: list[float]) -> float:
    gap = 0.0
    for node in range(len(own_exposure)):
        if other_exposure[node] > own_exposure[node]:
            gap += other_exposure[node] - own_exposure[node]
    return gap


def build_common_bias_start(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    precomputed,
) -> SparseSolution:
    campaign1: set[int] = set()
    campaign2: set[int] = set()

    while len(campaign1) + len(campaign2) + 2 <= budget:
        state = build_search_state_from_solution(
            graph=graph,
            initial_seeds=initial_seed_sets,
            precomputed=precomputed,
            balanced_seed_sets=SeedSets(campaign1=sorted(campaign1), campaign2=sorted(campaign2)),
        )
        common_pool = build_common_candidate_pool(
            graph=graph,
            precomputed=precomputed,
            initial_campaign1=set(initial_seed_sets.campaign1),
            initial_campaign2=set(initial_seed_sets.campaign2),
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        )
        if not common_pool:
            break
        best_node = max(
            common_pool[:candidate_frontier_size(graph)],
            key=lambda node: common_structural_rank(precomputed, node),
        )
        campaign1.add(best_node)
        campaign2.add(best_node)

    if len(campaign1) + len(campaign2) < budget:
        state = build_search_state_from_solution(
            graph=graph,
            initial_seeds=initial_seed_sets,
            precomputed=precomputed,
            balanced_seed_sets=SeedSets(campaign1=sorted(campaign1), campaign2=sorted(campaign2)),
        )
        pool1 = build_campaign_candidate_pool(
            graph=graph,
            campaign=1,
            precomputed=precomputed,
            initial_campaign_nodes=set(initial_seed_sets.campaign1),
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        )
        pool2 = build_campaign_candidate_pool(
            graph=graph,
            campaign=2,
            precomputed=precomputed,
            initial_campaign_nodes=set(initial_seed_sets.campaign2),
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        )
        best1 = None
        best2 = None
        score_add1 = float("-inf")
        score_add2 = float("-inf")
        if pool1:
            best1 = max(
                pool1[:candidate_frontier_size(graph)],
                key=lambda node: campaign_structural_rank(precomputed, 1, node) + exposure_gap_bonus(state, 1, node),
            )
            score_add1 = campaign_structural_rank(precomputed, 1, best1) + exposure_gap_bonus(state, 1, best1)
        if pool2:
            best2 = max(
                pool2[:candidate_frontier_size(graph)],
                key=lambda node: campaign_structural_rank(precomputed, 2, node) + exposure_gap_bonus(state, 2, node),
            )
            score_add2 = campaign_structural_rank(precomputed, 2, best2) + exposure_gap_bonus(state, 2, best2)
        if best1 is not None or best2 is not None:
            if score_add1 >= score_add2:
                if best1 is not None:
                    campaign1.add(best1)
            elif best2 is not None:
                campaign2.add(best2)

    return SparseSolution(campaign1=campaign1, campaign2=campaign2)


def build_repair_bias_start(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    precomputed,
) -> SparseSolution:
    campaign1: set[int] = set()
    campaign2: set[int] = set()

    while len(campaign1) + len(campaign2) < budget:
        state = build_search_state_from_solution(
            graph=graph,
            initial_seeds=initial_seed_sets,
            precomputed=precomputed,
            balanced_seed_sets=SeedSets(campaign1=sorted(campaign1), campaign2=sorted(campaign2)),
        )
        gap1 = total_gap(state.approx1, state.approx2)
        gap2 = total_gap(state.approx2, state.approx1)
        prefer_campaign = 1 if gap1 >= gap2 else 2

        pool1 = build_campaign_candidate_pool(
            graph=graph,
            campaign=1,
            precomputed=precomputed,
            initial_campaign_nodes=set(initial_seed_sets.campaign1),
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        )
        pool2 = build_campaign_candidate_pool(
            graph=graph,
            campaign=2,
            precomputed=precomputed,
            initial_campaign_nodes=set(initial_seed_sets.campaign2),
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        )

        best1 = None
        best2 = None
        score_add1 = float("-inf")
        score_add2 = float("-inf")
        if pool1:
            best1 = max(
                pool1[:candidate_frontier_size(graph)],
                key=lambda node: campaign_structural_rank(precomputed, 1, node) + exposure_gap_bonus(state, 1, node),
            )
            score_add1 = campaign_structural_rank(precomputed, 1, best1) + exposure_gap_bonus(state, 1, best1)
        if pool2:
            best2 = max(
                pool2[:candidate_frontier_size(graph)],
                key=lambda node: campaign_structural_rank(precomputed, 2, node) + exposure_gap_bonus(state, 2, node),
            )
            score_add2 = campaign_structural_rank(precomputed, 2, best2) + exposure_gap_bonus(state, 2, best2)

        if best1 is None and best2 is None:
            break

        if prefer_campaign == 1:
            if best1 is not None and (best2 is None or score_add1 >= score_add2 - 1e-9):
                campaign1.add(best1)
            elif best2 is not None:
                campaign2.add(best2)
        else:
            if best2 is not None and (best1 is None or score_add2 >= score_add1 - 1e-9):
                campaign2.add(best2)
            elif best1 is not None:
                campaign1.add(best1)

    return SparseSolution(campaign1=campaign1, campaign2=campaign2)


def build_profile_start_candidates(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    precomputed,
    score1: list[float],
    score2: list[float],
    timer: Timer,
) -> list[SparseSolution]:
    weaker_campaign = initial_strength_gap(score1, score2, initial_seed_sets)
    common_quota = budget // 2
    mixed_common = max(1, budget // 4)
    remainder = budget % 2

    if weaker_campaign == 1:
        heavy_profile = SplitProfile(
            s1_only=budget - 2 * mixed_common,
            s2_only=0,
            both=mixed_common,
        )
        common_profile = SplitProfile(s1_only=remainder, s2_only=0, both=common_quota)
    else:
        heavy_profile = SplitProfile(
            s1_only=0,
            s2_only=budget - 2 * mixed_common,
            both=mixed_common,
        )
        common_profile = SplitProfile(s1_only=0, s2_only=remainder, both=common_quota)

    profiles = [
        SplitProfile(s1_only=budget // 2, s2_only=budget - budget // 2, both=0),
        common_profile,
        heavy_profile,
    ]

    starts: list[SparseSolution] = []
    seen_profiles: set[tuple[int, int, int]] = set()
    for profile in profiles:
        key = (profile.s1_only, profile.s2_only, profile.both)
        if key in seen_profiles:
            continue
        seen_profiles.add(key)
        if not timer.has_time(profile_start_min_seconds(graph)):
            break
        try:
            state = construct_solution_for_profile(
                graph=graph,
                initial_seeds=initial_seed_sets,
                precomputed=precomputed,
                profile=profile,
                budget=budget,
                timer=timer,
            )
        except TimeoutError:
            break
        starts.append(
            SparseSolution(campaign1=set(state.selected1_set), campaign2=set(state.selected2_set))
        )
    return starts


def perturb_start_solution(
    graph,
    initial_seed_sets: SeedSets,
    source: SparseSolution,
    budget: int,
    precomputed,
    ranked1: list[int],
    ranked2: list[int],
    score1: list[float],
    score2: list[float],
    rng,
) -> SparseSolution:
    current = source.copy()
    for _ in range(perturb_steps(graph)):
        if precomputed is not None:
            neighbors = build_structured_neighbors(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                current=current,
                budget=budget,
                precomputed=precomputed,
                rng=rng,
            )
        else:
            neighbors = build_basic_neighbor_batch(
                graph=graph,
                current=current,
                budget=budget,
                ranked1=ranked1,
                ranked2=ranked2,
                score1=score1,
                score2=score2,
                rng=rng,
            )
        if not neighbors:
            break
        current = neighbors[rng.randrange(len(neighbors))].copy()
    return current


def build_start_solutions(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    baseline_sparse: SparseSolution,
    warm_sparse: SparseSolution,
    precomputed,
    score1: list[float],
    score2: list[float],
    ranked1: list[int],
    ranked2: list[int],
    timer: Timer,
    base_seed: int,
) -> list[SparseSolution]:
    target_starts = restart_count(graph)
    starts: dict[tuple[tuple[int, ...], tuple[int, ...]], SparseSolution] = {}
    register_start_solution(starts, baseline_sparse)
    register_start_solution(starts, warm_sparse)

    if target_starts <= 1:
        return [warm_sparse]

    rng = RandomContext(seed=base_seed + 1703).py_random

    if precomputed is not None and timer.has_time(start_generation_min_seconds(graph)):
        register_start_solution(
            starts,
            build_common_bias_start(graph, initial_seed_sets, budget, precomputed),
        )
        register_start_solution(
            starts,
            build_repair_bias_start(graph, initial_seed_sets, budget, precomputed),
        )
        for candidate in build_profile_start_candidates(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            budget=budget,
            precomputed=precomputed,
            score1=score1,
            score2=score2,
            timer=timer,
        ):
            register_start_solution(starts, candidate)

    existing = list(starts.values())
    if existing:
        register_start_solution(
            starts,
            perturb_start_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                source=warm_sparse,
                budget=budget,
                precomputed=precomputed,
                ranked1=ranked1,
                ranked2=ranked2,
                score1=score1,
                score2=score2,
                rng=rng,
            ),
        )
        strongest_source = max(existing, key=lambda item: item.total_size)
        register_start_solution(
            starts,
            perturb_start_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                source=strongest_source,
                budget=budget,
                precomputed=precomputed,
                ranked1=ranked1,
                ranked2=ranked2,
                score1=score1,
                score2=score2,
                rng=rng,
            ),
        )

    return list(starts.values())


def run_annealing_trajectory(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    start_sparse: SparseSolution,
    precomputed,
    ranked1: list[int],
    ranked2: list[int],
    score1: list[float],
    score2: list[float],
    fast_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    accurate_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    fast_eval_seed: int,
    accurate_eval_seed: int,
    timer: Timer,
    attempt_seed: int,
) -> tuple[SparseSolution, float]:
    rng = RandomContext(seed=attempt_seed).py_random
    current_sparse = start_sparse.copy()
    current_score = accurate_eval_solution(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=current_sparse,
        score_cache=accurate_cache,
        eval_seed=accurate_eval_seed,
    )
    local_best_sparse = current_sparse.copy()
    local_best_score = current_score
    temperature = initial_temperature(graph, current_score)
    cooling = cooling_rate(graph)
    no_improve_steps = 0
    reheated = False

    for iteration in range(iterations_per_start(graph)):
        if not timer.has_time(NEIGHBOR_MIN_SECONDS):
            break

        if precomputed is not None:
            candidate_batch = build_structured_neighbors(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                current=current_sparse,
                budget=budget,
                precomputed=precomputed,
                rng=rng,
            )
        else:
            candidate_batch = build_basic_neighbor_batch(
                graph=graph,
                current=current_sparse,
                budget=budget,
                ranked1=ranked1,
                ranked2=ranked2,
                score1=score1,
                score2=score2,
                rng=rng,
            )

        if not candidate_batch:
            break

        fast_ranked: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SparseSolution]] = []
        for candidate_sparse in candidate_batch:
            fast_score = fast_eval_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                solution=candidate_sparse,
                score_cache=fast_cache,
                eval_seed=fast_eval_seed,
            )
            fast_ranked.append((fast_score, sparse_signature(candidate_sparse), candidate_sparse))

        if not fast_ranked:
            break

        fast_ranked.sort(key=lambda item: (-item[0], item[1]))
        elite_count = min(len(fast_ranked), accurate_eval_keep(graph))
        accurate_ranked: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SparseSolution]] = []
        for _, signature, candidate_sparse in fast_ranked[:elite_count]:
            if not timer.has_time(accurate_eval_min_seconds(graph)):
                break
            accurate_score = accurate_eval_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                solution=candidate_sparse,
                score_cache=accurate_cache,
                eval_seed=accurate_eval_seed,
            )
            accurate_ranked.append((accurate_score, signature, candidate_sparse))

        if not accurate_ranked:
            break

        accurate_ranked.sort(key=lambda item: (-item[0], item[1]))
        top_score, _, top_sparse = accurate_ranked[0]
        if top_score > local_best_score + 1e-9:
            local_best_sparse = top_sparse.copy()
            local_best_score = top_score
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        candidate_score, _, candidate_sparse = choose_transition_candidate(accurate_ranked, rng)
        delta = candidate_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
            current_sparse = candidate_sparse
            current_score = candidate_score

        if no_improve_steps >= stall_limit(graph):
            if not reheated and timer.has_time(accurate_eval_min_seconds(graph) + NEIGHBOR_MIN_SECONDS):
                reheated = True
                no_improve_steps = 0
                current_sparse = local_best_sparse.copy()
                current_score = local_best_score
                temperature = max(temperature, initial_temperature(graph, local_best_score) * reheat_multiplier(graph))
            else:
                break

        temperature = max(0.1, temperature * cooling)

    return local_best_sparse, local_best_score


def run_last_mile_refine(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    start_sparse: SparseSolution,
    precomputed,
    ranked1: list[int],
    ranked2: list[int],
    score1: list[float],
    score2: list[float],
    fast_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    accurate_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    fast_eval_seed: int,
    accurate_eval_seed: int,
    timer: Timer,
    attempt_seed: int,
) -> tuple[SparseSolution, float]:
    rng = RandomContext(seed=attempt_seed).py_random
    base_sparse = start_sparse.copy()
    base_score = accurate_eval_solution(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=base_sparse,
        score_cache=accurate_cache,
        eval_seed=accurate_eval_seed,
    )
    current_sparse = base_sparse.copy()
    current_score = base_score
    best_sparse = base_sparse.copy()
    best_score = base_score

    if timer.has_time(last_mile_min_seconds(graph) + accurate_eval_min_seconds(graph)):
        perturbed_sparse = perturb_start_solution(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            source=base_sparse,
            budget=budget,
            precomputed=precomputed,
            ranked1=ranked1,
            ranked2=ranked2,
            score1=score1,
            score2=score2,
            rng=rng,
        )
        perturbed_score = accurate_eval_solution(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            solution=perturbed_sparse,
            score_cache=accurate_cache,
            eval_seed=accurate_eval_seed,
        )
        if perturbed_score > best_score:
            best_sparse = perturbed_sparse.copy()
            best_score = perturbed_score
        current_sparse = perturbed_sparse
        current_score = perturbed_score

    temperature = max(0.5, initial_temperature(graph, max(best_score, current_score)) * 0.4)
    cooling = 0.92 if graph.num_nodes <= SMALL_GRAPH_THRESHOLD else 0.96

    for _ in range(last_mile_steps(graph)):
        if not timer.has_time(last_mile_min_seconds(graph)):
            break

        if precomputed is not None:
            candidate_batch = build_structured_neighbors(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                current=current_sparse,
                budget=budget,
                precomputed=precomputed,
                rng=rng,
            )
        else:
            candidate_batch = build_basic_neighbor_batch(
                graph=graph,
                current=current_sparse,
                budget=budget,
                ranked1=ranked1,
                ranked2=ranked2,
                score1=score1,
                score2=score2,
                rng=rng,
            )

        if not candidate_batch:
            break

        fast_ranked: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SparseSolution]] = []
        for candidate_sparse in candidate_batch:
            fast_score = fast_eval_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                solution=candidate_sparse,
                score_cache=fast_cache,
                eval_seed=fast_eval_seed,
            )
            fast_ranked.append((fast_score, sparse_signature(candidate_sparse), candidate_sparse))

        if not fast_ranked:
            break

        fast_ranked.sort(key=lambda item: (-item[0], item[1]))
        shortlist = fast_ranked[: last_mile_fast_keep(graph)]

        improved_sparse: SparseSolution | None = None
        improved_score = best_score
        accurate_ranked: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SparseSolution]] = []
        for _, signature, candidate_sparse in shortlist[: last_mile_accurate_keep(graph)]:
            if not timer.has_time(accurate_eval_min_seconds(graph)):
                break
            accurate_score = accurate_eval_solution(
                graph=graph,
                initial_seed_sets=initial_seed_sets,
                solution=candidate_sparse,
                score_cache=accurate_cache,
                eval_seed=accurate_eval_seed,
            )
            accurate_ranked.append((accurate_score, signature, candidate_sparse))
            if accurate_score > improved_score + 1e-9:
                improved_sparse = candidate_sparse.copy()
                improved_score = accurate_score

        if not accurate_ranked:
            break

        accurate_ranked.sort(key=lambda item: (-item[0], item[1]))
        candidate_score, _, candidate_sparse = choose_transition_candidate(accurate_ranked, rng)
        delta = candidate_score - current_score
        if delta >= 0.0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
            current_sparse = candidate_sparse.copy()
            current_score = candidate_score

        if improved_sparse is not None:
            best_sparse = improved_sparse
            best_score = improved_score

        temperature = max(0.1, temperature * cooling)

    return best_sparse, best_score


def heuristic_warm_start(
    graph,
    initial_seed_sets: SeedSets,
    budget: int,
    timer: Timer,
) -> SeedSets:
    baseline_solution = build_guard_baseline_solution(graph, initial_seed_sets, budget)
    allowed_seconds = warm_start_time_limit(graph, timer)
    if allowed_seconds < 5.0:
        return baseline_solution

    previous_limit = os.environ.get("IEMP_HEUR_TIME_LIMIT_SECONDS")
    previous_reserve = os.environ.get("IEMP_HEUR_TIME_RESERVE_SECONDS")
    os.environ["IEMP_HEUR_TIME_LIMIT_SECONDS"] = str(allowed_seconds)
    os.environ["IEMP_HEUR_TIME_RESERVE_SECONDS"] = "3"

    try:
        return build_heuristic_solution(graph, initial_seed_sets, budget)
    except Exception:
        return baseline_solution
    finally:
        if previous_limit is None:
            os.environ.pop("IEMP_HEUR_TIME_LIMIT_SECONDS", None)
        else:
            os.environ["IEMP_HEUR_TIME_LIMIT_SECONDS"] = previous_limit
        if previous_reserve is None:
            os.environ.pop("IEMP_HEUR_TIME_RESERVE_SECONDS", None)
        else:
            os.environ["IEMP_HEUR_TIME_RESERVE_SECONDS"] = previous_reserve


def build_evolutionary_solution(graph, initial_seed_sets: SeedSets, budget: int) -> SeedSets:
    timer = create_evol_timer(graph)
    baseline_solution = build_timeout_baseline_solution(graph, initial_seed_sets, budget)
    baseline_sparse = sparse_from_seed_sets(baseline_solution)
    fast_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    accurate_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    base_seed = 20401000 + graph.num_nodes * 131 + graph.num_edges * 17 + budget
    fast_eval_seed = base_seed + 311
    accurate_eval_seed = base_seed + 911

    best_sparse = baseline_sparse
    best_score = accurate_eval_solution(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=baseline_sparse,
        score_cache=accurate_cache,
        eval_seed=accurate_eval_seed,
    )

    if not timer.has_time(NEIGHBOR_MIN_SECONDS):
        return build_guard_baseline_solution(graph, initial_seed_sets, budget)

    warm_seed_sets = heuristic_warm_start(graph, initial_seed_sets, budget, timer)
    current_sparse = sparse_from_seed_sets(warm_seed_sets)
    current_score = accurate_eval_solution(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        solution=current_sparse,
        score_cache=accurate_cache,
        eval_seed=accurate_eval_seed,
    )

    if current_score > best_score:
        best_sparse = current_sparse.copy()
        best_score = current_score

    score1, score2 = compute_campaign_scores(graph)
    ranked1 = build_ranked_candidates(score1, set(initial_seed_sets.campaign1))
    ranked2 = build_ranked_candidates(score2, set(initial_seed_sets.campaign2))
    precomputed = None

    if timer.has_time(precompute_min_seconds(graph)):
        try:
            precomputed = build_precomputed_data(graph, initial_seed_sets, timer=timer)
        except TimeoutError:
            precomputed = None

    start_solutions = build_start_solutions(
        graph=graph,
        initial_seed_sets=initial_seed_sets,
        budget=budget,
        baseline_sparse=baseline_sparse,
        warm_sparse=current_sparse,
        precomputed=precomputed,
        score1=score1,
        score2=score2,
        ranked1=ranked1,
        ranked2=ranked2,
        timer=timer,
        base_seed=base_seed,
    )

    ranked_starts: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SparseSolution]] = []
    for start_sparse in start_solutions:
        start_score = accurate_eval_solution(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            solution=start_sparse,
            score_cache=accurate_cache,
            eval_seed=accurate_eval_seed,
        )
        ranked_starts.append((start_score, sparse_signature(start_sparse), start_sparse))

    ranked_starts.sort(key=lambda item: (-item[0], item[1]))
    selected_starts = ranked_starts[: restart_count(graph)]

    for attempt_index, (_, _, start_sparse) in enumerate(selected_starts):
        if not timer.has_time(NEIGHBOR_MIN_SECONDS):
            break
        local_best_sparse, local_best_score = run_annealing_trajectory(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            budget=budget,
            start_sparse=start_sparse,
            precomputed=precomputed,
            ranked1=ranked1,
            ranked2=ranked2,
            score1=score1,
            score2=score2,
            fast_cache=fast_cache,
            accurate_cache=accurate_cache,
            fast_eval_seed=fast_eval_seed,
            accurate_eval_seed=accurate_eval_seed,
            timer=timer,
            attempt_seed=base_seed + 3001 + attempt_index * 97,
        )
        if local_best_score > best_score:
            best_sparse = local_best_sparse.copy()
            best_score = local_best_score

    if timer.has_time(last_mile_min_seconds(graph)):
        last_mile_sparse, last_mile_score = run_last_mile_refine(
            graph=graph,
            initial_seed_sets=initial_seed_sets,
            budget=budget,
            start_sparse=best_sparse,
            precomputed=precomputed,
            ranked1=ranked1,
            ranked2=ranked2,
            score1=score1,
            score2=score2,
            fast_cache=fast_cache,
            accurate_cache=accurate_cache,
            fast_eval_seed=fast_eval_seed,
            accurate_eval_seed=accurate_eval_seed,
            timer=timer,
            attempt_seed=base_seed + 900001,
        )
        if last_mile_score > best_score:
            best_sparse = last_mile_sparse.copy()
            best_score = last_mile_score

    return best_sparse.to_seed_sets()


def main() -> None:
    args = parse_common_args(needs_output_path=False)
    graph = None
    initial_seed_sets: SeedSets | None = None
    try:
        graph = load_graph(args.network_path)
        initial_seed_sets = load_seed_sets(args.initial_seed_path)
        validate_seed_sets(initial_seed_sets, num_nodes=graph.num_nodes)
        try:
            balanced_seed_sets = build_evolutionary_solution(graph, initial_seed_sets, args.budget)
        except TimeoutError:
            balanced_seed_sets = build_guard_baseline_solution(graph, initial_seed_sets, args.budget)
        except Exception:
            balanced_seed_sets = build_submission_fallback_solution(graph, initial_seed_sets, args.budget)
        validate_seed_sets(balanced_seed_sets, num_nodes=graph.num_nodes, budget=args.budget)
    except Exception:
        balanced_seed_sets = build_submission_fallback_solution(graph, initial_seed_sets, args.budget)
    write_seed_sets(args.balanced_seed_path, balanced_seed_sets)


if __name__ == "__main__":
    main()
