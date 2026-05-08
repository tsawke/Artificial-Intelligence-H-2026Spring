from __future__ import annotations

from collections import deque
import heapq
import os
from dataclasses import dataclass

from Evaluator import (
    RandomContext,
    SeedSets,
    Timer,
    load_graph,
    load_seed_sets,
    merged_seed_list,
    parse_common_args,
    validate_seed_sets,
    write_seed_sets,
)


SMALL_GRAPH_THRESHOLD = 1024
OPENING_STEPS_SMALL = 3
OPENING_STEPS_LARGE = 1
POOL_LIMIT_SMALL = 180
POOL_LIMIT_LARGE = 240
COMMON_POOL_LIMIT_SMALL = 160
COMMON_POOL_LIMIT_LARGE = 220
BASE_QUOTA = 64
COVER_QUOTA = 64
COMMON_QUOTA = 48
NEARBY_QUOTA = 48
REPAIR_TARGET_QUOTA = 36
REPAIR_SOURCE_QUOTA = 96
CHEAP_TOP_SMALL = 24
CHEAP_TOP_LARGE = 24
SHARED_STAGE_SAMPLES_SMALL = (8, 12, 20)
SHARED_STAGE_SAMPLES_LARGE = (8, 10, 12)
PROFILE_KEEP_SMALL = 8
PROFILE_KEEP_LARGE = 4
PROFILE_EVAL_WORLDS_SMALL = 24
PROFILE_EVAL_WORLDS_LARGE = 10
BEAM_WIDTH_SMALL = 3
BEAM_WIDTH_LARGE = 2
BEAM_DEPTH_SMALL = 3
BEAM_DEPTH_LARGE = 2
BEAM_BRANCH_COUNT = 2
LOCAL_ITER_SMALL = 3
LOCAL_ITER_LARGE = 1
LOCAL_OUT_LIMIT_SMALL = 3
LOCAL_OUT_LIMIT_LARGE = 2
LOCAL_IN_LIMIT_SMALL = 5
LOCAL_IN_LIMIT_LARGE = 3
LOCAL_APPROX_KEEP_SMALL = 10
LOCAL_APPROX_KEEP_LARGE = 6
LOCAL_EVAL_KEEP_SMALL = 4
LOCAL_EVAL_KEEP_LARGE = 3
LOCAL_QUICK_WORLDS_SMALL = 12
LOCAL_QUICK_WORLDS_LARGE = 4
LOCAL_TABU_SIZE_SMALL = 10
LOCAL_TABU_SIZE_LARGE = 6
LAST_MILE_ITER_SMALL = 1
LAST_MILE_ITER_LARGE = 1
LAST_MILE_OUT_LIMIT_SMALL = 2
LAST_MILE_OUT_LIMIT_LARGE = 1
LAST_MILE_IN_LIMIT_SMALL = 3
LAST_MILE_IN_LIMIT_LARGE = 2
LAST_MILE_APPROX_KEEP_SMALL = 6
LAST_MILE_APPROX_KEEP_LARGE = 4
LAST_MILE_EVAL_KEEP_SMALL = 3
LAST_MILE_EVAL_KEEP_LARGE = 2
LAST_MILE_QUICK_WORLDS_SMALL = 6
LAST_MILE_QUICK_WORLDS_LARGE = 2
LAST_MILE_TABU_SIZE_SMALL = 4
LAST_MILE_TABU_SIZE_LARGE = 3
LOCAL_DENSE_OUTDEG_THRESHOLD = 8.0
PRECOMPUTE_CHECK_INTERVAL = 64
PROFILE_MIN_SECONDS_SMALL = 4.0
PROFILE_MIN_SECONDS_LARGE = 18.0
LAST_MILE_MIN_SECONDS_SMALL = 0.25
LAST_MILE_MIN_SECONDS_LARGE = 3.0
LOCAL_SEARCH_MIN_SECONDS_SMALL = 3.0
LOCAL_SEARCH_MIN_SECONDS_LARGE = 12.0
RERANK_MIN_SECONDS = 0.15
BASELINE_BALANCE_PENALTY = 1.2
BASELINE_COMMON_ACTION_BONUS = 0.12
BASELINE_MIRROR_ACTION_BONUS = 0.06
BASELINE_REFINEMENT_POOL = 12
BASELINE_REFINEMENT_WEAK = 3
BASELINE_REFINEMENT_ITERS = 2
BASELINE_GUARD_RESERVE_SCALE = 1.25


@dataclass(slots=True)
class PrecomputedHeuristicData:
    direct_nodes: list[list[int]]
    second_nodes1: list[list[int]]
    second_probs1: list[list[float]]
    second_nodes2: list[list[int]]
    second_probs2: list[list[float]]
    base_score1: list[float]
    base_score2: list[float]
    coverage_strength1: list[float]
    coverage_strength2: list[float]
    common_strength: list[float]
    combined_strength: list[float]
    ranked_base1: list[int]
    ranked_base2: list[int]
    ranked_cover1: list[int]
    ranked_cover2: list[int]
    ranked_common: list[int]
    ranked_combined: list[int]
    nearby_ranked1: list[int]
    nearby_ranked2: list[int]


@dataclass(slots=True)
class CandidateAction:
    campaign: int
    node: int
    cheap_score: float
    priority: float


@dataclass(slots=True, frozen=True)
class SplitProfile:
    s1_only: int
    s2_only: int
    both: int


@dataclass(slots=True)
class SearchState:
    selected1: list[int]
    selected2: list[int]
    selected1_set: set[int]
    selected2_set: set[int]
    approx1: list[float]
    approx2: list[float]
    strength1: float
    strength2: float
    used_budget: int
    action_index: int
    count_s1_only: int
    count_s2_only: int
    count_both: int
    approx_score: float
    beam_score: float
    lazy_cache: dict[tuple[int, int], float]


@dataclass(slots=True)
class SharedWorld:
    live_masks1: list[int]
    live_masks2: list[int]
    active1: bytearray
    reached1: bytearray
    active2: bytearray
    reached2: bytearray


class IncrementalCampaignWorkspace:
    __slots__ = ("active", "reached", "queue", "touched_active", "touched_reached")

    def __init__(self, num_nodes: int) -> None:
        self.active = bytearray(num_nodes)
        self.reached = bytearray(num_nodes)
        self.queue: list[int] = []
        self.touched_active: list[int] = []
        self.touched_reached: list[int] = []

    def reset(self) -> None:
        for node in self.touched_active:
            self.active[node] = 0
        for node in self.touched_reached:
            self.reached[node] = 0
        self.queue.clear()
        self.touched_active.clear()
        self.touched_reached.clear()


class IncrementalEvalWorkspace:
    __slots__ = ("campaign1", "campaign2", "marked", "union_nodes")

    def __init__(self, num_nodes: int) -> None:
        self.campaign1 = IncrementalCampaignWorkspace(num_nodes)
        self.campaign2 = IncrementalCampaignWorkspace(num_nodes)
        self.marked = bytearray(num_nodes)
        self.union_nodes: list[int] = []


def action_cost(campaign: int) -> int:
    return 2 if campaign == 3 else 1


def action_priority(score: float, campaign: int) -> float:
    return score / action_cost(campaign)


def profile_total_budget(profile: SplitProfile) -> int:
    return profile.s1_only + profile.s2_only + 2 * profile.both


def action_profile_available(profile: SplitProfile, state: SearchState, campaign: int) -> bool:
    if campaign == 1:
        return state.count_s1_only < profile.s1_only
    if campaign == 2:
        return state.count_s2_only < profile.s2_only
    return state.count_both < profile.both


def compute_campaign_scores(graph) -> tuple[list[float], list[float]]:
    score1 = [0.0] * graph.num_nodes
    score2 = [0.0] * graph.num_nodes

    for node in range(graph.num_nodes):
        out_degree = len(graph.out_neighbors[node])
        score1[node] = 1.0 + out_degree + graph.out_weight_sum1[node]
        score2[node] = 1.0 + out_degree + graph.out_weight_sum2[node]

    return score1, score2


def build_ranked_candidates(
    scores: list[float],
    forbidden_same_campaign: set[int],
) -> list[int]:
    candidates = [node for node in range(len(scores)) if node not in forbidden_same_campaign]
    candidates.sort(key=lambda node: (-scores[node], node))
    return candidates


def next_available(
    ranked_nodes: list[int],
    pointer: int,
    selected1: set[int],
    selected2: set[int],
) -> tuple[int | None, int]:
    while pointer < len(ranked_nodes):
        node = ranked_nodes[pointer]
        if node not in selected1 and node not in selected2:
            return node, pointer
        pointer += 1
    return None, pointer


def current_strength(scores: list[float], nodes: list[int]) -> float:
    return sum(scores[node] for node in nodes)


def choose_opening_action(
    ranked1: list[int],
    ranked2: list[int],
    pointer1: int,
    pointer2: int,
    selected1: set[int],
    selected2: set[int],
    score1: list[float],
    score2: list[float],
    strength1: float,
    strength2: float,
) -> tuple[int | None, int | None, int, int]:
    candidate1, pointer1 = next_available(ranked1, pointer1, selected1, selected2)
    candidate2, pointer2 = next_available(ranked2, pointer2, selected1, selected2)

    if candidate1 is None and candidate2 is None:
        return None, None, pointer1, pointer2
    if candidate1 is None:
        return 2, candidate2, pointer1, pointer2 + 1
    if candidate2 is None:
        return 1, candidate1, pointer1 + 1, pointer2

    diff_if_add_1 = abs((strength1 + score1[candidate1]) - strength2)
    diff_if_add_2 = abs(strength1 - (strength2 + score2[candidate2]))

    if diff_if_add_1 < diff_if_add_2:
        return 1, candidate1, pointer1 + 1, pointer2
    if diff_if_add_2 < diff_if_add_1:
        return 2, candidate2, pointer1, pointer2 + 1
    if score1[candidate1] >= score2[candidate2]:
        return 1, candidate1, pointer1 + 1, pointer2
    return 2, candidate2, pointer1, pointer2 + 1


def compute_baseline_common_scores(score1: list[float], score2: list[float]) -> list[float]:
    return [
        min(score1[node], score2[node]) + 0.25 * (score1[node] + score2[node])
        for node in range(len(score1))
    ]


def build_baseline_common_ranked(common_scores: list[float], forbidden_nodes: set[int]) -> list[int]:
    ranked = [node for node in range(len(common_scores)) if node not in forbidden_nodes]
    ranked.sort(key=lambda node: (-common_scores[node], node))
    return ranked


def baseline_proxy_value(strength1: float, strength2: float) -> float:
    return strength1 + strength2 - BASELINE_BALANCE_PENALTY * abs(strength1 - strength2)


def remove_node_once(nodes: list[int], node: int) -> None:
    for index, value in enumerate(nodes):
        if value == node:
            nodes.pop(index)
            return
    raise ValueError(f"Node {node} not found in selected list.")


def choose_best_mirror_candidate(
    source_nodes: list[int],
    target_selected: set[int],
    target_initial: set[int],
    target_scores: list[float],
    common_scores: list[float],
) -> int | None:
    best_node: int | None = None
    best_key: tuple[float, float, int] | None = None
    for node in source_nodes:
        if node in target_selected or node in target_initial:
            continue
        key = (target_scores[node], common_scores[node], -node)
        if best_key is None or key > best_key:
            best_key = key
            best_node = node
    return best_node


def choose_timeout_baseline_action(
    ranked1: list[int],
    ranked2: list[int],
    ranked_common: list[int],
    pointer1: int,
    pointer2: int,
    common_pointer: int,
    selected1: list[int],
    selected2: list[int],
    selected1_set: set[int],
    selected2_set: set[int],
    initial1_set: set[int],
    initial2_set: set[int],
    score1: list[float],
    score2: list[float],
    common_scores: list[float],
    strength1: float,
    strength2: float,
    remaining_budget: int,
) -> tuple[str, int, int, int, int, float] | None:
    current_value = baseline_proxy_value(strength1, strength2)
    actions: list[tuple[float, float, float, int, str, int, int, int, int]] = []

    candidate1, next_pointer1 = next_available(ranked1, pointer1, selected1_set, selected2_set)
    if candidate1 is not None:
        gain = baseline_proxy_value(strength1 + score1[candidate1], strength2) - current_value
        actions.append(
            (gain, gain, score1[candidate1], 1, "add1", candidate1, next_pointer1 + 1, pointer2, common_pointer)
        )

    candidate2, next_pointer2 = next_available(ranked2, pointer2, selected1_set, selected2_set)
    if candidate2 is not None:
        gain = baseline_proxy_value(strength1, strength2 + score2[candidate2]) - current_value
        actions.append(
            (gain, gain, score2[candidate2], 1, "add2", candidate2, pointer1, next_pointer2 + 1, common_pointer)
        )

    mirror_to1 = choose_best_mirror_candidate(selected2, selected1_set, initial1_set, score1, common_scores)
    if mirror_to1 is not None:
        gain = (
            baseline_proxy_value(strength1 + score1[mirror_to1], strength2)
            - current_value
            + BASELINE_MIRROR_ACTION_BONUS * common_scores[mirror_to1]
        )
        actions.append((gain, gain, common_scores[mirror_to1], 1, "mirror1", mirror_to1, pointer1, pointer2, common_pointer))

    mirror_to2 = choose_best_mirror_candidate(selected1, selected2_set, initial2_set, score2, common_scores)
    if mirror_to2 is not None:
        gain = (
            baseline_proxy_value(strength1, strength2 + score2[mirror_to2])
            - current_value
            + BASELINE_MIRROR_ACTION_BONUS * common_scores[mirror_to2]
        )
        actions.append((gain, gain, common_scores[mirror_to2], 1, "mirror2", mirror_to2, pointer1, pointer2, common_pointer))

    if remaining_budget >= 2:
        common_node, next_common_pointer = next_available(ranked_common, common_pointer, selected1_set, selected2_set)
        if common_node is not None:
            raw_gain = baseline_proxy_value(
                strength1 + score1[common_node],
                strength2 + score2[common_node],
            ) - current_value
            gain = raw_gain + BASELINE_COMMON_ACTION_BONUS * common_scores[common_node]
            actions.append(
                (
                    gain / 2.0,
                    gain,
                    common_scores[common_node],
                    2,
                    "both",
                    common_node,
                    pointer1,
                    pointer2,
                    next_common_pointer + 1,
                )
            )

    if not actions:
        return None

    _, gain, _, _, kind, node, new_pointer1, new_pointer2, new_common_pointer = max(actions)
    return kind, node, new_pointer1, new_pointer2, new_common_pointer, gain


def weakest_single_nodes(
    selected_nodes: list[int],
    opposite_selected: set[int],
    scores: list[float],
    common_scores: list[float],
    limit: int,
) -> list[int]:
    single_nodes = [node for node in selected_nodes if node not in opposite_selected]
    single_nodes.sort(key=lambda node: (scores[node] + 0.2 * common_scores[node], node))
    return single_nodes[:limit]


def top_fresh_nodes(
    ranked_nodes: list[int],
    selected1: set[int],
    selected2: set[int],
    limit: int,
) -> list[int]:
    fresh: list[int] = []
    for node in ranked_nodes:
        if node in selected1 or node in selected2:
            continue
        fresh.append(node)
        if len(fresh) >= limit:
            break
    return fresh


def refine_guard_baseline_solution(
    selected1: list[int],
    selected2: list[int],
    selected1_set: set[int],
    selected2_set: set[int],
    initial1_set: set[int],
    initial2_set: set[int],
    ranked1: list[int],
    ranked2: list[int],
    ranked_common: list[int],
    score1: list[float],
    score2: list[float],
    common_scores: list[float],
    strength1: float,
    strength2: float,
) -> tuple[list[int], list[int], float, float]:
    for _ in range(BASELINE_REFINEMENT_ITERS):
        current_value = baseline_proxy_value(strength1, strength2)
        weak1 = weakest_single_nodes(selected1, selected2_set, score1, common_scores, BASELINE_REFINEMENT_WEAK)
        weak2 = weakest_single_nodes(selected2, selected1_set, score2, common_scores, BASELINE_REFINEMENT_WEAK)
        fresh1 = top_fresh_nodes(ranked1, selected1_set, selected2_set, BASELINE_REFINEMENT_POOL)
        fresh2 = top_fresh_nodes(ranked2, selected1_set, selected2_set, BASELINE_REFINEMENT_POOL)
        fresh_common = top_fresh_nodes(ranked_common, selected1_set, selected2_set, BASELINE_REFINEMENT_POOL)

        best_action: tuple[float, str, tuple[int, ...]] | None = None

        for removed in weak1:
            for added in fresh1:
                gain = baseline_proxy_value(strength1 - score1[removed] + score1[added], strength2) - current_value
                candidate = (gain, "swap1", (removed, added))
                if best_action is None or candidate > best_action:
                    best_action = candidate

        for removed in weak2:
            for added in fresh2:
                gain = baseline_proxy_value(strength1, strength2 - score2[removed] + score2[added]) - current_value
                candidate = (gain, "swap2", (removed, added))
                if best_action is None or candidate > best_action:
                    best_action = candidate

        only1_ranked = sorted(
            (node for node in selected1 if node not in selected2_set and node not in initial2_set),
            key=lambda node: (-common_scores[node], -score2[node], node),
        )[:BASELINE_REFINEMENT_POOL]
        only2_ranked = sorted(
            (node for node in selected2 if node not in selected1_set and node not in initial1_set),
            key=lambda node: (-common_scores[node], -score1[node], node),
        )[:BASELINE_REFINEMENT_POOL]

        for removed in weak2:
            for mirrored in only1_ranked:
                gain = (
                    baseline_proxy_value(strength1, strength2 - score2[removed] + score2[mirrored]) - current_value
                    + 0.08 * common_scores[mirrored]
                )
                candidate = (gain, "mirror_to2", (removed, mirrored))
                if best_action is None or candidate > best_action:
                    best_action = candidate

        for removed in weak1:
            for mirrored in only2_ranked:
                gain = (
                    baseline_proxy_value(strength1 - score1[removed] + score1[mirrored], strength2) - current_value
                    + 0.08 * common_scores[mirrored]
                )
                candidate = (gain, "mirror_to1", (removed, mirrored))
                if best_action is None or candidate > best_action:
                    best_action = candidate

        for removed1 in weak1:
            for removed2 in weak2:
                for common_node in fresh_common:
                    gain = (
                        baseline_proxy_value(
                            strength1 - score1[removed1] + score1[common_node],
                            strength2 - score2[removed2] + score2[common_node],
                        )
                        - current_value
                        + 0.16 * common_scores[common_node]
                    )
                    candidate = (gain, "pair_to_common", (removed1, removed2, common_node))
                    if best_action is None or candidate > best_action:
                        best_action = candidate

        if best_action is None or best_action[0] <= 1e-9:
            break

        _, action_kind, payload = best_action
        if action_kind == "swap1":
            removed, added = payload
            remove_node_once(selected1, removed)
            selected1_set.remove(removed)
            selected1.append(added)
            selected1_set.add(added)
            strength1 += score1[added] - score1[removed]
            continue
        if action_kind == "swap2":
            removed, added = payload
            remove_node_once(selected2, removed)
            selected2_set.remove(removed)
            selected2.append(added)
            selected2_set.add(added)
            strength2 += score2[added] - score2[removed]
            continue
        if action_kind == "mirror_to2":
            removed, mirrored = payload
            remove_node_once(selected2, removed)
            selected2_set.remove(removed)
            selected2.append(mirrored)
            selected2_set.add(mirrored)
            strength2 += score2[mirrored] - score2[removed]
            continue
        if action_kind == "mirror_to1":
            removed, mirrored = payload
            remove_node_once(selected1, removed)
            selected1_set.remove(removed)
            selected1.append(mirrored)
            selected1_set.add(mirrored)
            strength1 += score1[mirrored] - score1[removed]
            continue

        removed1, removed2, common_node = payload
        remove_node_once(selected1, removed1)
        remove_node_once(selected2, removed2)
        selected1_set.remove(removed1)
        selected2_set.remove(removed2)
        selected1.append(common_node)
        selected2.append(common_node)
        selected1_set.add(common_node)
        selected2_set.add(common_node)
        strength1 += score1[common_node] - score1[removed1]
        strength2 += score2[common_node] - score2[removed2]

    return selected1, selected2, strength1, strength2


def build_direct_nodes(graph, timer: Timer | None = None) -> list[list[int]]:
    direct_nodes: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    seen_stamp = [0] * graph.num_nodes
    stamp = 0

    for node in range(graph.num_nodes):
        if timer is not None and node % PRECOMPUTE_CHECK_INTERVAL == 0:
            timer.checkpoint()
        stamp += 1
        nodes = [node]
        seen_stamp[node] = stamp
        for neighbor in graph.out_neighbors[node]:
            if seen_stamp[neighbor] == stamp:
                continue
            seen_stamp[neighbor] = stamp
            nodes.append(neighbor)
        direct_nodes[node] = nodes

    return direct_nodes


def build_second_hop_info(
    graph,
    first_edge_probs: list[list[float]],
    direct_nodes: list[list[int]],
    timer: Timer | None = None,
) -> tuple[list[list[int]], list[list[float]], list[float]]:
    second_nodes: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    second_probs: list[list[float]] = [[] for _ in range(graph.num_nodes)]
    coverage_strength = [0.0] * graph.num_nodes

    for node in range(graph.num_nodes):
        if timer is not None and node % PRECOMPUTE_CHECK_INTERVAL == 0:
            timer.checkpoint()
        blocked = set(direct_nodes[node])
        aggregated: dict[int, float] = {}
        neighbors = graph.out_neighbors[node]
        probs = first_edge_probs[node]

        for idx, middle in enumerate(neighbors):
            activate_prob = probs[idx]
            if activate_prob <= 0.0:
                continue
            for target in graph.out_neighbors[middle]:
                if target in blocked:
                    continue
                previous = aggregated.get(target, 0.0)
                aggregated[target] = 1.0 - (1.0 - previous) * (1.0 - activate_prob)

        nodes = list(aggregated.keys())
        probs_list = [aggregated[target] for target in nodes]
        second_nodes[node] = nodes
        second_probs[node] = probs_list
        coverage_strength[node] = len(direct_nodes[node]) + sum(probs_list)

    return second_nodes, second_probs, coverage_strength


def build_nearby_ranked_nodes(graph, seeds: list[int], score: list[float]) -> list[int]:
    nearby: set[int] = set()
    seed_set = set(seeds)

    for seed in seeds:
        nearby.add(seed)
        for neighbor in graph.out_neighbors[seed]:
            nearby.add(neighbor)
            for second in graph.out_neighbors[neighbor]:
                nearby.add(second)
        for parent in graph.in_neighbors[seed]:
            nearby.add(parent)
            for grand_parent in graph.in_neighbors[parent]:
                nearby.add(grand_parent)

    ranked = [node for node in nearby if node not in seed_set]
    ranked.sort(key=lambda node: (-score[node], node))
    return ranked


def build_precomputed_data(
    graph,
    initial_seeds: SeedSets,
    timer: Timer | None = None,
) -> PrecomputedHeuristicData:
    direct_nodes = build_direct_nodes(graph, timer=timer)
    base_score1, base_score2 = compute_campaign_scores(graph)
    second_nodes1, second_probs1, coverage_strength1 = build_second_hop_info(
        graph,
        graph.out_prob1,
        direct_nodes,
        timer=timer,
    )
    second_nodes2, second_probs2, coverage_strength2 = build_second_hop_info(
        graph,
        graph.out_prob2,
        direct_nodes,
        timer=timer,
    )
    common_strength = [
        min(coverage_strength1[node], coverage_strength2[node]) for node in range(graph.num_nodes)
    ]
    combined_strength = [
        coverage_strength1[node] + coverage_strength2[node] for node in range(graph.num_nodes)
    ]
    ranked_base1 = build_ranked_candidates(base_score1, set(initial_seeds.campaign1))
    ranked_base2 = build_ranked_candidates(base_score2, set(initial_seeds.campaign2))
    ranked_cover1 = build_ranked_candidates(coverage_strength1, set(initial_seeds.campaign1))
    ranked_cover2 = build_ranked_candidates(coverage_strength2, set(initial_seeds.campaign2))
    ranked_common = sorted(range(graph.num_nodes), key=lambda node: (-common_strength[node], node))
    ranked_combined = sorted(range(graph.num_nodes), key=lambda node: (-combined_strength[node], node))
    nearby_ranked1 = build_nearby_ranked_nodes(graph, initial_seeds.campaign1, coverage_strength1)
    nearby_ranked2 = build_nearby_ranked_nodes(graph, initial_seeds.campaign2, coverage_strength2)

    return PrecomputedHeuristicData(
        direct_nodes=direct_nodes,
        second_nodes1=second_nodes1,
        second_probs1=second_probs1,
        second_nodes2=second_nodes2,
        second_probs2=second_probs2,
        base_score1=base_score1,
        base_score2=base_score2,
        coverage_strength1=coverage_strength1,
        coverage_strength2=coverage_strength2,
        common_strength=common_strength,
        combined_strength=combined_strength,
        ranked_base1=ranked_base1,
        ranked_base2=ranked_base2,
        ranked_cover1=ranked_cover1,
        ranked_cover2=ranked_cover2,
        ranked_common=ranked_common,
        ranked_combined=ranked_combined,
        nearby_ranked1=nearby_ranked1,
        nearby_ranked2=nearby_ranked2,
    )


def apply_approx_update(
    approx_exposure: list[float],
    direct_nodes: list[int],
    second_nodes: list[int],
    second_probs: list[float],
) -> None:
    for node in direct_nodes:
        approx_exposure[node] = 1.0

    for idx, node in enumerate(second_nodes):
        prob = second_probs[idx]
        approx_exposure[node] = 1.0 - (1.0 - approx_exposure[node]) * (1.0 - prob)


def initialize_approx_exposure(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
) -> tuple[list[float], list[float]]:
    approx1 = [0.0] * graph.num_nodes
    approx2 = [0.0] * graph.num_nodes

    for seed in initial_seeds.campaign1:
        apply_approx_update(
            approx1,
            precomputed.direct_nodes[seed],
            precomputed.second_nodes1[seed],
            precomputed.second_probs1[seed],
        )

    for seed in initial_seeds.campaign2:
        apply_approx_update(
            approx2,
            precomputed.direct_nodes[seed],
            precomputed.second_nodes2[seed],
            precomputed.second_probs2[seed],
        )

    return approx1, approx2


def approximate_balanced_score(approx1: list[float], approx2: list[float]) -> float:
    total = 0.0
    for idx in range(len(approx1)):
        total += 1.0 - abs(approx1[idx] - approx2[idx])
    return total


def build_prefix_sum(ranked_nodes: list[int], values: list[float], limit: int) -> list[float]:
    prefix = [0.0]
    running = 0.0
    for node in ranked_nodes[:limit]:
        running += values[node]
        prefix.append(running)
    return prefix


def prefix_value(prefix: list[float], count: int) -> float:
    capped = min(count, len(prefix) - 1)
    return prefix[capped]


def enumerate_split_profiles(budget: int) -> list[SplitProfile]:
    profiles: list[SplitProfile] = []
    for both_count in range((budget // 2) + 1):
        remaining_budget = budget - 2 * both_count
        for total_single in range(remaining_budget + 1):
            for s1_only in range(total_single + 1):
                s2_only = total_single - s1_only
                profiles.append(SplitProfile(s1_only=s1_only, s2_only=s2_only, both=both_count))
    return profiles


def select_top_profiles(
    graph,
    precomputed: PrecomputedHeuristicData,
    initial_seeds: SeedSets,
    budget: int,
) -> list[SplitProfile]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        keep_count = PROFILE_KEEP_SMALL
    else:
        keep_count = PROFILE_KEEP_LARGE

    prefix_cover1 = build_prefix_sum(precomputed.ranked_cover1, precomputed.coverage_strength1, budget)
    prefix_cover2 = build_prefix_sum(precomputed.ranked_cover2, precomputed.coverage_strength2, budget)
    prefix_common = build_prefix_sum(precomputed.ranked_common, precomputed.common_strength, budget)
    base1 = current_strength(precomputed.coverage_strength1, initial_seeds.campaign1)
    base2 = current_strength(precomputed.coverage_strength2, initial_seeds.campaign2)

    scored_profiles: list[tuple[float, SplitProfile]] = []
    for profile in enumerate_split_profiles(budget):
        s1_gain = prefix_value(prefix_cover1, profile.s1_only)
        s2_gain = prefix_value(prefix_cover2, profile.s2_only)
        both_gain = prefix_value(prefix_common, profile.both)
        est1 = base1 + s1_gain + both_gain
        est2 = base2 + s2_gain + both_gain
        score = min(est1, est2) - 0.05 * abs(est1 - est2) + 0.01 * (est1 + est2)
        scored_profiles.append((score, profile))

    scored_profiles.sort(
        key=lambda item: (
            -item[0],
            -item[1].both,
            -min(item[1].s1_only, item[1].s2_only),
            -max(item[1].s1_only, item[1].s2_only),
        )
    )

    weaker_campaign = 1 if base1 <= base2 else 2
    selected_profiles: list[SplitProfile] = []
    seen_profiles: set[SplitProfile] = set()

    def add_profile(profile: SplitProfile | None) -> None:
        if profile is None or profile in seen_profiles:
            return
        seen_profiles.add(profile)
        selected_profiles.append(profile)

    def first_matching(predicate) -> SplitProfile | None:
        for _, profile in scored_profiles:
            if predicate(profile):
                return profile
        return None

    max_both = max(profile.both for _, profile in scored_profiles)
    unrestricted_profile = SplitProfile(s1_only=budget, s2_only=budget, both=budget // 2)
    mid_common = max(1, budget // 3)
    if weaker_campaign == 1:
        weaker_repair_profile = SplitProfile(
            s1_only=max(0, budget - 2 * mid_common),
            s2_only=0,
            both=mid_common,
        )
    else:
        weaker_repair_profile = SplitProfile(
            s1_only=0,
            s2_only=max(0, budget - 2 * mid_common),
            both=mid_common,
        )

    add_profile(unrestricted_profile)
    add_profile(scored_profiles[0][1] if scored_profiles else None)
    add_profile(first_matching(lambda profile: profile.both == max_both))
    add_profile(weaker_repair_profile)
    add_profile(first_matching(lambda profile: abs(profile.s1_only - profile.s2_only) <= 1))
    add_profile(first_matching(lambda profile: profile.both == 0))

    for _, profile in scored_profiles:
        add_profile(profile)
        if len(selected_profiles) >= keep_count:
            break

    return selected_profiles[:keep_count]


def infer_heuristic_time_limit_seconds(graph) -> float:
    if graph.num_nodes <= 1000:
        return 90.0
    if graph.num_nodes == 36742 and graph.num_edges == 49248:
        return 840.0
    if graph.num_nodes == 7115 and graph.num_edges == 103689:
        return 660.0
    if graph.num_nodes == 3454 and graph.num_edges == 32140:
        return 540.0

    average_out_degree = graph.num_edges / max(1, graph.num_nodes)
    if graph.num_nodes >= 20000:
        return 840.0
    if average_out_degree >= 10.0:
        return 660.0
    if graph.num_nodes >= 3000:
        return 540.0
    return 90.0


def infer_heuristic_time_reserve_seconds(time_limit_seconds: float) -> float:
    if time_limit_seconds <= 120.0:
        base_reserve = 8.0
    elif time_limit_seconds <= 600.0:
        base_reserve = 25.0
    else:
        base_reserve = 40.0
    return base_reserve * BASELINE_GUARD_RESERVE_SCALE


def create_heuristic_timer(graph) -> Timer:
    override_limit = os.getenv("IEMP_HEUR_TIME_LIMIT_SECONDS")
    override_reserve = os.getenv("IEMP_HEUR_TIME_RESERVE_SECONDS")

    if override_limit is not None:
        time_limit_seconds = float(override_limit)
    else:
        time_limit_seconds = infer_heuristic_time_limit_seconds(graph)

    if override_reserve is not None:
        reserve_seconds = float(override_reserve)
    else:
        reserve_seconds = infer_heuristic_time_reserve_seconds(time_limit_seconds)

    return Timer(time_limit_seconds=time_limit_seconds, reserve_seconds=reserve_seconds)


def profile_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return PROFILE_MIN_SECONDS_SMALL
    return PROFILE_MIN_SECONDS_LARGE


def local_search_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LOCAL_SEARCH_MIN_SECONDS_SMALL
    return LOCAL_SEARCH_MIN_SECONDS_LARGE


def last_mile_min_seconds(graph) -> float:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        return LAST_MILE_MIN_SECONDS_SMALL
    return LAST_MILE_MIN_SECONDS_LARGE


def build_timeout_baseline_solution(graph, initial_seeds: SeedSets, budget: int) -> SeedSets:
    score1, score2 = compute_campaign_scores(graph)
    ranked1 = build_ranked_candidates(score1, set(initial_seeds.campaign1))
    ranked2 = build_ranked_candidates(score2, set(initial_seeds.campaign2))

    selected1: list[int] = []
    selected2: list[int] = []
    selected1_set: set[int] = set()
    selected2_set: set[int] = set()
    pointer1 = 0
    pointer2 = 0
    used_budget = 0
    strength1 = current_strength(score1, initial_seeds.campaign1)
    strength2 = current_strength(score2, initial_seeds.campaign2)
    opening_steps = OPENING_STEPS_SMALL if graph.num_nodes <= SMALL_GRAPH_THRESHOLD else OPENING_STEPS_LARGE

    for _ in range(min(opening_steps, budget)):
        campaign, node, pointer1, pointer2 = choose_opening_action(
            ranked1=ranked1,
            ranked2=ranked2,
            pointer1=pointer1,
            pointer2=pointer2,
            selected1=selected1_set,
            selected2=selected2_set,
            score1=score1,
            score2=score2,
            strength1=strength1,
            strength2=strength2,
        )
        if campaign is None or node is None:
            break
        if campaign == 1:
            selected1.append(node)
            selected1_set.add(node)
            strength1 += score1[node]
        else:
            selected2.append(node)
            selected2_set.add(node)
            strength2 += score2[node]
        used_budget += 1

    while used_budget < budget:
        candidate1, pointer1 = next_available(ranked1, pointer1, selected1_set, selected2_set)
        candidate2, pointer2 = next_available(ranked2, pointer2, selected1_set, selected2_set)

        if candidate1 is None and candidate2 is None:
            break
        if candidate1 is None:
            selected2.append(candidate2)
            selected2_set.add(candidate2)
            strength2 += score2[candidate2]
            used_budget += 1
            pointer2 += 1
            continue
        if candidate2 is None:
            selected1.append(candidate1)
            selected1_set.add(candidate1)
            strength1 += score1[candidate1]
            used_budget += 1
            pointer1 += 1
            continue

        diff_if_add_1 = abs((strength1 + score1[candidate1]) - strength2)
        diff_if_add_2 = abs(strength1 - (strength2 + score2[candidate2]))
        if diff_if_add_1 < diff_if_add_2 or (
            diff_if_add_1 == diff_if_add_2 and score1[candidate1] >= score2[candidate2]
        ):
            selected1.append(candidate1)
            selected1_set.add(candidate1)
            strength1 += score1[candidate1]
            pointer1 += 1
        else:
            selected2.append(candidate2)
            selected2_set.add(candidate2)
            strength2 += score2[candidate2]
            pointer2 += 1
        used_budget += 1

    return SeedSets(campaign1=selected1, campaign2=selected2)


def build_guard_baseline_solution(graph, initial_seeds: SeedSets, budget: int) -> SeedSets:
    score1, score2 = compute_campaign_scores(graph)
    ranked1 = build_ranked_candidates(score1, set(initial_seeds.campaign1))
    ranked2 = build_ranked_candidates(score2, set(initial_seeds.campaign2))
    initial1_set = set(initial_seeds.campaign1)
    initial2_set = set(initial_seeds.campaign2)
    forbidden_common = set(initial1_set)
    forbidden_common.update(initial2_set)
    common_scores = compute_baseline_common_scores(score1, score2)
    ranked_common = build_baseline_common_ranked(common_scores, forbidden_common)

    selected1: list[int] = []
    selected2: list[int] = []
    selected1_set: set[int] = set()
    selected2_set: set[int] = set()
    pointer1 = 0
    pointer2 = 0
    common_pointer = 0
    used_budget = 0
    strength1 = current_strength(score1, initial_seeds.campaign1)
    strength2 = current_strength(score2, initial_seeds.campaign2)
    opening_steps = OPENING_STEPS_SMALL if graph.num_nodes <= SMALL_GRAPH_THRESHOLD else OPENING_STEPS_LARGE

    for _ in range(min(opening_steps, budget)):
        campaign, node, pointer1, pointer2 = choose_opening_action(
            ranked1=ranked1,
            ranked2=ranked2,
            pointer1=pointer1,
            pointer2=pointer2,
            selected1=selected1_set,
            selected2=selected2_set,
            score1=score1,
            score2=score2,
            strength1=strength1,
            strength2=strength2,
        )
        if campaign is None or node is None:
            break
        if campaign == 1:
            selected1.append(node)
            selected1_set.add(node)
            strength1 += score1[node]
        else:
            selected2.append(node)
            selected2_set.add(node)
            strength2 += score2[node]
        used_budget += 1

    while used_budget < budget:
        chosen_action = choose_timeout_baseline_action(
            ranked1=ranked1,
            ranked2=ranked2,
            ranked_common=ranked_common,
            pointer1=pointer1,
            pointer2=pointer2,
            common_pointer=common_pointer,
            selected1=selected1,
            selected2=selected2,
            selected1_set=selected1_set,
            selected2_set=selected2_set,
            initial1_set=initial1_set,
            initial2_set=initial2_set,
            score1=score1,
            score2=score2,
            common_scores=common_scores,
            strength1=strength1,
            strength2=strength2,
            remaining_budget=budget - used_budget,
        )
        if chosen_action is None:
            break
        action_kind, node, pointer1, pointer2, common_pointer, action_gain = chosen_action
        if action_gain <= 0.0 and used_budget > 0:
            break

        if action_kind == "add1":
            selected1.append(node)
            selected1_set.add(node)
            strength1 += score1[node]
            used_budget += 1
            continue
        if action_kind == "add2":
            selected2.append(node)
            selected2_set.add(node)
            strength2 += score2[node]
            used_budget += 1
            continue
        if action_kind == "mirror1":
            selected1.append(node)
            selected1_set.add(node)
            strength1 += score1[node]
            used_budget += 1
            continue
        if action_kind == "mirror2":
            selected2.append(node)
            selected2_set.add(node)
            strength2 += score2[node]
            used_budget += 1
            continue

        selected1.append(node)
        selected2.append(node)
        selected1_set.add(node)
        selected2_set.add(node)
        strength1 += score1[node]
        strength2 += score2[node]
        used_budget += 2

    selected1, selected2, _, _ = refine_guard_baseline_solution(
        selected1=selected1,
        selected2=selected2,
        selected1_set=selected1_set,
        selected2_set=selected2_set,
        initial1_set=initial1_set,
        initial2_set=initial2_set,
        ranked1=ranked1,
        ranked2=ranked2,
        ranked_common=ranked_common,
        score1=score1,
        score2=score2,
        common_scores=common_scores,
        strength1=strength1,
        strength2=strength2,
    )

    return SeedSets(campaign1=selected1, campaign2=selected2)


def extend_from_ranked(
    destination: set[int],
    ranked_nodes: list[int],
    limit: int,
    invalid_nodes: set[int],
) -> None:
    added = 0
    for node in ranked_nodes:
        if node in invalid_nodes or node in destination:
            continue
        destination.add(node)
        added += 1
        if added >= limit:
            return


def select_top_gap_targets(own_exposure: list[float], other_exposure: list[float], limit: int) -> list[int]:
    positive_gap_nodes = [
        (other_exposure[node] - own_exposure[node], node)
        for node in range(len(own_exposure))
        if other_exposure[node] - own_exposure[node] > 1e-9
    ]
    positive_gap_nodes.sort(key=lambda item: (-item[0], item[1]))
    return [node for _, node in positive_gap_nodes[:limit]]


def select_top_dual_gap_targets(approx1: list[float], approx2: list[float], limit: int) -> list[int]:
    half_limit = max(1, limit // 2)
    candidates = select_top_gap_targets(approx1, approx2, half_limit)
    candidates.extend(select_top_gap_targets(approx2, approx1, half_limit))
    seen: set[int] = set()
    ordered: list[int] = []
    for node in candidates:
        if node in seen:
            continue
        seen.add(node)
        ordered.append(node)
    return ordered[:limit]


def collect_repair_source_nodes(
    graph,
    targets: list[int],
    invalid_nodes: set[int],
    static_score: list[float],
    limit: int,
) -> list[int]:
    candidates: set[int] = set()
    soft_cap = max(limit * 4, limit)

    for target in targets:
        if target not in invalid_nodes:
            candidates.add(target)
        for parent in graph.in_neighbors[target]:
            if parent not in invalid_nodes:
                candidates.add(parent)
            for grand_parent in graph.in_neighbors[parent]:
                if grand_parent not in invalid_nodes:
                    candidates.add(grand_parent)
            if len(candidates) >= soft_cap:
                break
        if len(candidates) >= soft_cap:
            break

    ranked = list(candidates)
    ranked.sort(key=lambda node: (-static_score[node], node))
    return ranked[:limit]


def build_campaign_candidate_pool(
    graph,
    campaign: int,
    precomputed: PrecomputedHeuristicData,
    initial_campaign_nodes: set[int],
    selected1: set[int],
    selected2: set[int],
    approx1: list[float],
    approx2: list[float],
) -> list[int]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        pool_limit = POOL_LIMIT_SMALL
    else:
        pool_limit = POOL_LIMIT_LARGE

    if campaign == 1:
        ranked_base = precomputed.ranked_base1
        ranked_cover = precomputed.ranked_cover1
        nearby_ranked = precomputed.nearby_ranked1
        static_score = precomputed.coverage_strength1
        own_exposure = approx1
        other_exposure = approx2
    else:
        ranked_base = precomputed.ranked_base2
        ranked_cover = precomputed.ranked_cover2
        nearby_ranked = precomputed.nearby_ranked2
        static_score = precomputed.coverage_strength2
        own_exposure = approx2
        other_exposure = approx1

    invalid_nodes = set(initial_campaign_nodes)
    invalid_nodes.update(selected1)
    invalid_nodes.update(selected2)

    pool: set[int] = set()
    extend_from_ranked(pool, ranked_base, BASE_QUOTA, invalid_nodes)
    extend_from_ranked(pool, ranked_cover, COVER_QUOTA, invalid_nodes)
    extend_from_ranked(pool, precomputed.ranked_common, COMMON_QUOTA, invalid_nodes)
    extend_from_ranked(pool, nearby_ranked, NEARBY_QUOTA, invalid_nodes)

    repair_targets = select_top_gap_targets(own_exposure, other_exposure, REPAIR_TARGET_QUOTA)
    repair_ranked = collect_repair_source_nodes(
        graph,
        repair_targets,
        invalid_nodes,
        static_score,
        REPAIR_SOURCE_QUOTA,
    )
    extend_from_ranked(pool, repair_ranked, REPAIR_SOURCE_QUOTA, invalid_nodes)

    if len(pool) <= pool_limit:
        ranked_pool = list(pool)
        ranked_pool.sort(key=lambda node: (-(static_score[node] + precomputed.common_strength[node]), node))
        return ranked_pool

    ranked_pool = list(pool)
    ranked_pool.sort(key=lambda node: (-(static_score[node] + precomputed.common_strength[node]), node))
    return ranked_pool[:pool_limit]


def build_common_candidate_pool(
    graph,
    precomputed: PrecomputedHeuristicData,
    initial_campaign1: set[int],
    initial_campaign2: set[int],
    selected1: set[int],
    selected2: set[int],
    approx1: list[float],
    approx2: list[float],
) -> list[int]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        pool_limit = COMMON_POOL_LIMIT_SMALL
    else:
        pool_limit = COMMON_POOL_LIMIT_LARGE

    invalid_nodes = set(initial_campaign1)
    invalid_nodes.update(initial_campaign2)
    invalid_nodes.update(selected1)
    invalid_nodes.update(selected2)

    pool: set[int] = set()
    extend_from_ranked(pool, precomputed.ranked_common, COMMON_QUOTA + 32, invalid_nodes)
    extend_from_ranked(pool, precomputed.ranked_combined, COVER_QUOTA, invalid_nodes)
    extend_from_ranked(pool, precomputed.nearby_ranked1, NEARBY_QUOTA // 2, invalid_nodes)
    extend_from_ranked(pool, precomputed.nearby_ranked2, NEARBY_QUOTA // 2, invalid_nodes)

    repair_targets = select_top_dual_gap_targets(approx1, approx2, REPAIR_TARGET_QUOTA)
    repair_ranked = collect_repair_source_nodes(
        graph,
        repair_targets,
        invalid_nodes,
        precomputed.combined_strength,
        REPAIR_SOURCE_QUOTA,
    )
    extend_from_ranked(pool, repair_ranked, REPAIR_SOURCE_QUOTA, invalid_nodes)

    ranked_pool = list(pool)
    ranked_pool.sort(
        key=lambda node: (-(precomputed.common_strength[node] + 0.6 * precomputed.combined_strength[node]), node)
    )
    return ranked_pool[:pool_limit]


def build_action_update_map(
    precomputed: PrecomputedHeuristicData,
    campaign: int,
    node: int,
) -> dict[int, float]:
    updates: dict[int, float] = {}

    for target in precomputed.direct_nodes[node]:
        updates[target] = 1.0

    if campaign == 1:
        second_nodes = precomputed.second_nodes1[node]
        second_probs = precomputed.second_probs1[node]
    else:
        second_nodes = precomputed.second_nodes2[node]
        second_probs = precomputed.second_probs2[node]

    for idx, target in enumerate(second_nodes):
        previous = updates.get(target, 0.0)
        updates[target] = 1.0 - (1.0 - previous) * (1.0 - second_probs[idx])

    return updates


def cheap_score_action(
    precomputed: PrecomputedHeuristicData,
    campaign: int,
    node: int,
    approx1: list[float],
    approx2: list[float],
) -> float:
    updates1: dict[int, float] = {}
    updates2: dict[int, float] = {}

    if campaign == 1:
        updates1 = build_action_update_map(precomputed, 1, node)
    elif campaign == 2:
        updates2 = build_action_update_map(precomputed, 2, node)
    else:
        updates1 = build_action_update_map(precomputed, 1, node)
        updates2 = build_action_update_map(precomputed, 2, node)

    union_nodes = set(updates1.keys())
    union_nodes.update(updates2.keys())

    balance_gain = 0.0
    coverage_gain = 0.0
    proximity_penalty = 0.0

    for target in union_nodes:
        old1 = approx1[target]
        old2 = approx2[target]
        update1 = updates1.get(target, 0.0)
        update2 = updates2.get(target, 0.0)
        new1 = 1.0 - (1.0 - old1) * (1.0 - update1)
        new2 = 1.0 - (1.0 - old2) * (1.0 - update2)

        balance_gain += abs(old1 - old2) - abs(new1 - new2)
        coverage_gain += (new1 - old1) + (new2 - old2)
        proximity_penalty += old1 * update1 + old2 * update2

    if campaign == 1:
        common_bonus = 0.12 * precomputed.common_strength[node]
        spread_bonus = 0.08 * precomputed.coverage_strength1[node] + 0.05 * precomputed.base_score1[node]
        cost_penalty = 0.0
    elif campaign == 2:
        common_bonus = 0.12 * precomputed.common_strength[node]
        spread_bonus = 0.08 * precomputed.coverage_strength2[node] + 0.05 * precomputed.base_score2[node]
        cost_penalty = 0.0
    else:
        common_bonus = 0.48 * precomputed.common_strength[node]
        spread_bonus = 0.06 * precomputed.combined_strength[node] + 0.03 * (
            precomputed.base_score1[node] + precomputed.base_score2[node]
        )
        cost_penalty = 0.5

    return 8.0 * balance_gain + 1.5 * coverage_gain + common_bonus + spread_bonus - 0.6 * proximity_penalty - cost_penalty


def static_action_priority(
    precomputed: PrecomputedHeuristicData,
    campaign: int,
    node: int,
) -> float:
    if campaign == 1:
        score = precomputed.coverage_strength1[node] + 0.25 * precomputed.common_strength[node]
    elif campaign == 2:
        score = precomputed.coverage_strength2[node] + 0.25 * precomputed.common_strength[node]
    else:
        score = 1.2 * precomputed.common_strength[node] + 0.5 * precomputed.combined_strength[node]
    return action_priority(score, campaign)


def collect_lazy_shortlist(
    precomputed: PrecomputedHeuristicData,
    approx1: list[float],
    approx2: list[float],
    candidate_keys: list[tuple[int, int]],
    lazy_cache: dict[tuple[int, int], float],
    target_count: int,
) -> list[CandidateAction]:
    heap: list[tuple[float, int, int]] = []
    for campaign, node in candidate_keys:
        estimate = lazy_cache.get((campaign, node))
        if estimate is None:
            estimate = static_action_priority(precomputed, campaign, node)
        heapq.heappush(heap, (-estimate, campaign, node))

    shortlist: list[CandidateAction] = []
    accepted: set[tuple[int, int]] = set()
    refresh_limit = max(target_count * 4, 24)
    refresh_count = 0

    while heap and len(shortlist) < target_count and refresh_count < refresh_limit:
        neg_estimate, campaign, node = heapq.heappop(heap)
        key = (campaign, node)
        refreshed_score = cheap_score_action(precomputed, campaign, node, approx1, approx2)
        refreshed_priority = action_priority(refreshed_score, campaign)
        lazy_cache[key] = refreshed_priority
        refresh_count += 1
        next_estimate = -heap[0][0] if heap else float("-inf")

        if refreshed_priority >= next_estimate - 1e-12:
            shortlist.append(
                CandidateAction(
                    campaign=campaign,
                    node=node,
                    cheap_score=refreshed_score,
                    priority=refreshed_priority,
                )
            )
            accepted.add(key)
        else:
            heapq.heappush(heap, (-refreshed_priority, campaign, node))

    while heap and len(shortlist) < target_count:
        _, campaign, node = heapq.heappop(heap)
        key = (campaign, node)
        if key in accepted:
            continue
        refreshed_score = cheap_score_action(precomputed, campaign, node, approx1, approx2)
        refreshed_priority = action_priority(refreshed_score, campaign)
        lazy_cache[key] = refreshed_priority
        shortlist.append(
            CandidateAction(
                campaign=campaign,
                node=node,
                cheap_score=refreshed_score,
                priority=refreshed_priority,
            )
        )

    shortlist.sort(key=lambda action: (-action.priority, -action.cheap_score, action.campaign, action.node))
    return shortlist


def sample_live_masks(graph, out_probs: list[list[float]], rng) -> list[int]:
    live_masks = [0] * graph.num_nodes
    for node in range(graph.num_nodes):
        mask = 0
        for idx, probability in enumerate(out_probs[node]):
            if rng.random() < probability:
                mask |= 1 << idx
        live_masks[node] = mask
    return live_masks


def simulate_state_in_live_world(
    graph,
    live_masks: list[int],
    seeds: list[int],
) -> tuple[bytearray, bytearray]:
    active = bytearray(graph.num_nodes)
    reached = bytearray(graph.num_nodes)
    queue: list[int] = []

    for seed in seeds:
        if not active[seed]:
            active[seed] = 1
            queue.append(seed)
        reached[seed] = 1

    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1
        neighbors = graph.out_neighbors[source]

        for target in neighbors:
            reached[target] = 1

        mask = live_masks[source]
        while mask:
            lowest_bit = mask & -mask
            edge_idx = lowest_bit.bit_length() - 1
            target = neighbors[edge_idx]
            if not active[target]:
                active[target] = 1
                reached[target] = 1
                queue.append(target)
            mask ^= lowest_bit

    return active, reached


def sample_shared_world(
    graph,
    campaign1_seeds: list[int],
    campaign2_seeds: list[int],
    rng,
) -> SharedWorld:
    live_masks1 = sample_live_masks(graph, graph.out_prob1, rng)
    live_masks2 = sample_live_masks(graph, graph.out_prob2, rng)
    active1, reached1 = simulate_state_in_live_world(graph, live_masks1, campaign1_seeds)
    active2, reached2 = simulate_state_in_live_world(graph, live_masks2, campaign2_seeds)
    return SharedWorld(
        live_masks1=live_masks1,
        live_masks2=live_masks2,
        active1=active1,
        reached1=reached1,
        active2=active2,
        reached2=reached2,
    )


def simulate_incremental_reached(
    graph,
    live_masks: list[int],
    base_active: bytearray,
    base_reached: bytearray,
    seed_nodes: list[int],
    workspace: IncrementalCampaignWorkspace,
) -> None:
    workspace.reset()
    active = workspace.active
    reached = workspace.reached
    queue = workspace.queue
    touched_active = workspace.touched_active
    touched_reached = workspace.touched_reached

    for seed in seed_nodes:
        if base_active[seed] or active[seed]:
            continue
        active[seed] = 1
        touched_active.append(seed)
        queue.append(seed)
        if not base_reached[seed]:
            reached[seed] = 1
            touched_reached.append(seed)

    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1
        neighbors = graph.out_neighbors[source]

        for target in neighbors:
            if not base_reached[target] and not reached[target]:
                reached[target] = 1
                touched_reached.append(target)

        mask = live_masks[source]
        while mask:
            lowest_bit = mask & -mask
            edge_idx = lowest_bit.bit_length() - 1
            target = neighbors[edge_idx]
            if not base_active[target] and not active[target]:
                active[target] = 1
                touched_active.append(target)
                queue.append(target)
                if not base_reached[target] and not reached[target]:
                    reached[target] = 1
                    touched_reached.append(target)
            mask ^= lowest_bit


def compute_balanced_delta(
    base_reached1: bytearray,
    base_reached2: bytearray,
    workspace: IncrementalEvalWorkspace,
) -> float:
    union_nodes = workspace.union_nodes
    union_nodes.clear()
    marked = workspace.marked
    added1 = workspace.campaign1.reached
    added2 = workspace.campaign2.reached

    for node in workspace.campaign1.touched_reached:
        if not marked[node]:
            marked[node] = 1
            union_nodes.append(node)
    for node in workspace.campaign2.touched_reached:
        if not marked[node]:
            marked[node] = 1
            union_nodes.append(node)

    delta = 0.0
    for node in union_nodes:
        old_mismatch = base_reached1[node] ^ base_reached2[node]
        new_mismatch = (base_reached1[node] or added1[node]) ^ (base_reached2[node] or added2[node])
        delta += old_mismatch - new_mismatch
        marked[node] = 0

    return delta


def evaluate_action_in_world(
    graph,
    world: SharedWorld,
    action: CandidateAction,
    workspace: IncrementalEvalWorkspace,
) -> float:
    if action.campaign == 1:
        simulate_incremental_reached(
            graph,
            world.live_masks1,
            world.active1,
            world.reached1,
            [action.node],
            workspace.campaign1,
        )
        workspace.campaign2.reset()
    elif action.campaign == 2:
        workspace.campaign1.reset()
        simulate_incremental_reached(
            graph,
            world.live_masks2,
            world.active2,
            world.reached2,
            [action.node],
            workspace.campaign2,
        )
    else:
        simulate_incremental_reached(
            graph,
            world.live_masks1,
            world.active1,
            world.reached1,
            [action.node],
            workspace.campaign1,
        )
        simulate_incremental_reached(
            graph,
            world.live_masks2,
            world.active2,
            world.reached2,
            [action.node],
            workspace.campaign2,
        )

    return compute_balanced_delta(world.reached1, world.reached2, workspace)


def rerank_top_actions_shared(
    graph,
    initial_seeds: SeedSets,
    selected1: list[int],
    selected2: list[int],
    actions: list[CandidateAction],
    step: int,
    timer: Timer | None = None,
) -> list[CandidateAction]:
    if not actions:
        return []

    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        stage_samples = SHARED_STAGE_SAMPLES_SMALL
    else:
        stage_samples = SHARED_STAGE_SAMPLES_LARGE

    survivors = sorted(actions, key=lambda action: (-action.priority, -action.cheap_score, action.campaign, action.node))
    cumulative_delta = {(action.campaign, action.node): 0.0 for action in survivors}
    total_worlds = 0
    current_seeds1 = merged_seed_list(initial_seeds.campaign1, selected1)
    current_seeds2 = merged_seed_list(initial_seeds.campaign2, selected2)
    rng = RandomContext(seed=20281000 + step * 104729).py_random
    workspace = IncrementalEvalWorkspace(graph.num_nodes)
    timeout_hit = False

    for stage_idx, world_count in enumerate(stage_samples):
        for _ in range(world_count):
            if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
                timeout_hit = True
                break
            world = sample_shared_world(graph, current_seeds1, current_seeds2, rng)
            total_worlds += 1
            for action in survivors:
                key = (action.campaign, action.node)
                cumulative_delta[key] += evaluate_action_in_world(graph, world, action, workspace)

        if timeout_hit:
            break

        if stage_idx == len(stage_samples) - 1 or len(survivors) == 1:
            continue

        next_size = max(1, (len(survivors) + 1) // 2)
        survivors.sort(
            key=lambda action: (
                -(cumulative_delta[(action.campaign, action.node)] / total_worlds) / action_cost(action.campaign),
                -action.priority,
                -action.cheap_score,
                -action.node,
            )
        )
        survivors = survivors[:next_size]

    if total_worlds == 0:
        return sorted(
            survivors,
            key=lambda action: (-action.priority, -action.cheap_score, action.campaign, action.node),
        )

    final_ranked = sorted(
        survivors,
        key=lambda action: (
            -(cumulative_delta[(action.campaign, action.node)] / total_worlds) / action_cost(action.campaign),
            -action.priority,
            -action.cheap_score,
            action.node,
        ),
    )

    reranked: list[CandidateAction] = []
    for action in final_ranked:
        average_gain = cumulative_delta[(action.campaign, action.node)] / total_worlds
        reranked.append(
            CandidateAction(
                campaign=action.campaign,
                node=action.node,
                cheap_score=action.cheap_score,
                priority=average_gain / action_cost(action.campaign),
            )
        )
    return reranked


def apply_action(
    precomputed: PrecomputedHeuristicData,
    node: int,
    campaign: int,
    selected1: list[int],
    selected2: list[int],
    selected1_set: set[int],
    selected2_set: set[int],
    approx1: list[float],
    approx2: list[float],
    strength1: float,
    strength2: float,
) -> tuple[float, float]:
    if campaign in (1, 3):
        selected1.append(node)
        selected1_set.add(node)
        strength1 += precomputed.base_score1[node]
        apply_approx_update(
            approx1,
            precomputed.direct_nodes[node],
            precomputed.second_nodes1[node],
            precomputed.second_probs1[node],
        )

    if campaign in (2, 3):
        selected2.append(node)
        selected2_set.add(node)
        strength2 += precomputed.base_score2[node]
        apply_approx_update(
            approx2,
            precomputed.direct_nodes[node],
            precomputed.second_nodes2[node],
            precomputed.second_probs2[node],
        )

    return strength1, strength2


def clone_state(state: SearchState) -> SearchState:
    return SearchState(
        selected1=state.selected1.copy(),
        selected2=state.selected2.copy(),
        selected1_set=set(state.selected1_set),
        selected2_set=set(state.selected2_set),
        approx1=state.approx1.copy(),
        approx2=state.approx2.copy(),
        strength1=state.strength1,
        strength2=state.strength2,
        used_budget=state.used_budget,
        action_index=state.action_index,
        count_s1_only=state.count_s1_only,
        count_s2_only=state.count_s2_only,
        count_both=state.count_both,
        approx_score=state.approx_score,
        beam_score=state.beam_score,
        lazy_cache=state.lazy_cache.copy(),
    )


def create_initial_state(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
) -> SearchState:
    approx1, approx2 = initialize_approx_exposure(graph, initial_seeds, precomputed)
    return SearchState(
        selected1=[],
        selected2=[],
        selected1_set=set(),
        selected2_set=set(),
        approx1=approx1,
        approx2=approx2,
        strength1=current_strength(precomputed.base_score1, initial_seeds.campaign1),
        strength2=current_strength(precomputed.base_score2, initial_seeds.campaign2),
        used_budget=0,
        action_index=0,
        count_s1_only=0,
        count_s2_only=0,
        count_both=0,
        approx_score=approximate_balanced_score(approx1, approx2),
        beam_score=0.0,
        lazy_cache={},
    )


def apply_action_to_state(
    precomputed: PrecomputedHeuristicData,
    state: SearchState,
    action: CandidateAction,
) -> None:
    state.strength1, state.strength2 = apply_action(
        precomputed=precomputed,
        node=action.node,
        campaign=action.campaign,
        selected1=state.selected1,
        selected2=state.selected2,
        selected1_set=state.selected1_set,
        selected2_set=state.selected2_set,
        approx1=state.approx1,
        approx2=state.approx2,
        strength1=state.strength1,
        strength2=state.strength2,
    )
    if action.campaign == 1:
        state.count_s1_only += 1
    elif action.campaign == 2:
        state.count_s2_only += 1
    else:
        state.count_both += 1

    state.used_budget += action_cost(action.campaign)
    state.action_index += 1
    state.beam_score += action.priority * action_cost(action.campaign)
    state.approx_score = approximate_balanced_score(state.approx1, state.approx2)


def state_sort_key(state: SearchState) -> tuple[float, float, float, int]:
    return (
        state.approx_score,
        state.beam_score,
        min(state.strength1, state.strength2),
        -abs(len(state.selected1) - len(state.selected2)),
    )


def build_candidate_keys_for_state(
    graph,
    precomputed: PrecomputedHeuristicData,
    initial_seeds: SeedSets,
    state: SearchState,
    profile: SplitProfile,
    budget: int,
) -> list[tuple[int, int]]:
    remaining_budget = budget - state.used_budget
    initial_campaign1 = set(initial_seeds.campaign1)
    initial_campaign2 = set(initial_seeds.campaign2)
    candidate_keys: list[tuple[int, int]] = []

    if remaining_budget >= 1 and action_profile_available(profile, state, 1):
        for node in build_campaign_candidate_pool(
            graph,
            campaign=1,
            precomputed=precomputed,
            initial_campaign_nodes=initial_campaign1,
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        ):
            candidate_keys.append((1, node))

    if remaining_budget >= 1 and action_profile_available(profile, state, 2):
        for node in build_campaign_candidate_pool(
            graph,
            campaign=2,
            precomputed=precomputed,
            initial_campaign_nodes=initial_campaign2,
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        ):
            candidate_keys.append((2, node))

    if remaining_budget >= 2 and action_profile_available(profile, state, 3):
        for node in build_common_candidate_pool(
            graph,
            precomputed=precomputed,
            initial_campaign1=initial_campaign1,
            initial_campaign2=initial_campaign2,
            selected1=state.selected1_set,
            selected2=state.selected2_set,
            approx1=state.approx1,
            approx2=state.approx2,
        ):
            candidate_keys.append((3, node))

    return candidate_keys


def rank_actions_for_state(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    state: SearchState,
    profile: SplitProfile,
    budget: int,
    desired_count: int,
    timer: Timer | None = None,
) -> list[CandidateAction]:
    if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
        return []

    candidate_keys = build_candidate_keys_for_state(
        graph=graph,
        initial_seeds=initial_seeds,
        precomputed=precomputed,
        state=state,
        profile=profile,
        budget=budget,
    )
    if not candidate_keys:
        return []

    shortlist_target = CHEAP_TOP_SMALL if graph.num_nodes <= SMALL_GRAPH_THRESHOLD else CHEAP_TOP_LARGE
    shortlist = collect_lazy_shortlist(
        precomputed=precomputed,
        approx1=state.approx1,
        approx2=state.approx2,
        candidate_keys=candidate_keys,
        lazy_cache=state.lazy_cache,
        target_count=shortlist_target,
    )
    reranked = rerank_top_actions_shared(
        graph=graph,
        initial_seeds=initial_seeds,
        selected1=state.selected1,
        selected2=state.selected2,
        actions=shortlist,
        step=state.action_index,
        timer=timer,
    )
    return reranked[:desired_count]


def complete_state_greedily(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    profile: SplitProfile,
    budget: int,
    state: SearchState,
    timer: Timer | None = None,
) -> SearchState:
    while state.used_budget < budget:
        if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
            break
        ranked_actions = rank_actions_for_state(
            graph=graph,
            initial_seeds=initial_seeds,
            precomputed=precomputed,
            state=state,
            profile=profile,
            budget=budget,
            desired_count=1,
            timer=timer,
        )
        if not ranked_actions:
            break
        apply_action_to_state(precomputed, state, ranked_actions[0])
    return state


def build_state_beam(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    profile: SplitProfile,
    budget: int,
    timer: Timer | None = None,
) -> list[SearchState]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        beam_width = BEAM_WIDTH_SMALL
        beam_depth = BEAM_DEPTH_SMALL
    else:
        beam_width = BEAM_WIDTH_LARGE
        beam_depth = BEAM_DEPTH_LARGE

    beam = [create_initial_state(graph, initial_seeds, precomputed)]

    for _ in range(beam_depth):
        if timer is not None and not timer.has_time(profile_min_seconds(graph)):
            break
        expanded_states: list[SearchState] = []
        expanded_any = False
        for state in beam:
            if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
                expanded_states.append(state)
                continue
            ranked_actions = rank_actions_for_state(
                graph=graph,
                initial_seeds=initial_seeds,
                precomputed=precomputed,
                state=state,
                profile=profile,
                budget=budget,
                desired_count=BEAM_BRANCH_COUNT,
                timer=timer,
            )
            if not ranked_actions:
                expanded_states.append(state)
                continue

            expanded_any = True
            for action in ranked_actions:
                child = clone_state(state)
                apply_action_to_state(precomputed, child, action)
                expanded_states.append(child)

        if not expanded_any:
            break

        deduplicated: dict[tuple[tuple[int, ...], tuple[int, ...]], SearchState] = {}
        for state in expanded_states:
            key = (tuple(sorted(state.selected1)), tuple(sorted(state.selected2)))
            existing = deduplicated.get(key)
            if existing is None or state_sort_key(state) > state_sort_key(existing):
                deduplicated[key] = state
        beam = sorted(deduplicated.values(), key=state_sort_key, reverse=True)[:beam_width]

    return beam


def count_balanced_from_reached(reached1: bytearray, reached2: bytearray) -> float:
    mismatch = 0
    for idx in range(len(reached1)):
        mismatch += reached1[idx] ^ reached2[idx]
    return float(len(reached1) - mismatch)


def estimate_solution_shared_worlds(
    graph,
    initial_seeds: SeedSets,
    balanced_seed_sets: SeedSets,
    num_worlds: int,
    random_seed: int,
) -> float:
    campaign1_seeds = merged_seed_list(initial_seeds.campaign1, balanced_seed_sets.campaign1)
    campaign2_seeds = merged_seed_list(initial_seeds.campaign2, balanced_seed_sets.campaign2)
    rng = RandomContext(seed=random_seed).py_random
    total = 0.0
    for _ in range(num_worlds):
        world = sample_shared_world(graph, campaign1_seeds, campaign2_seeds, rng)
        total += count_balanced_from_reached(world.reached1, world.reached2)
    return total / num_worlds


def construct_solution_for_profile(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    profile: SplitProfile,
    budget: int,
    timer: Timer | None = None,
) -> SearchState:
    beam = build_state_beam(
        graph=graph,
        initial_seeds=initial_seeds,
        precomputed=precomputed,
        profile=profile,
        budget=budget,
        timer=timer,
    )

    final_states: list[SearchState] = []
    for state in beam:
        if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
            break
        final_states.append(
            complete_state_greedily(
                graph=graph,
                initial_seeds=initial_seeds,
                precomputed=precomputed,
                profile=profile,
                budget=budget,
                state=clone_state(state),
                timer=timer,
            )
        )

    if timer is None or timer.has_time(RERANK_MIN_SECONDS):
        final_states.append(
            complete_state_greedily(
                graph=graph,
                initial_seeds=initial_seeds,
                precomputed=precomputed,
                profile=profile,
                budget=budget,
                state=create_initial_state(graph, initial_seeds, precomputed),
                timer=timer,
            )
        )

    if not final_states:
        return create_initial_state(graph, initial_seeds, precomputed)
    return max(final_states, key=state_sort_key)


def normalize_seed_sets(seed_sets: SeedSets) -> SeedSets:
    return SeedSets(
        campaign1=sorted(set(seed_sets.campaign1)),
        campaign2=sorted(set(seed_sets.campaign2)),
    )


def solution_signature(seed_sets: SeedSets) -> tuple[tuple[int, ...], tuple[int, ...]]:
    normalized = normalize_seed_sets(seed_sets)
    return (tuple(normalized.campaign1), tuple(normalized.campaign2))


def build_search_state_from_solution(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    balanced_seed_sets: SeedSets,
) -> SearchState:
    normalized = normalize_seed_sets(balanced_seed_sets)
    approx1, approx2 = initialize_approx_exposure(graph, initial_seeds, precomputed)

    for node in normalized.campaign1:
        apply_approx_update(
            approx1,
            precomputed.direct_nodes[node],
            precomputed.second_nodes1[node],
            precomputed.second_probs1[node],
        )
    for node in normalized.campaign2:
        apply_approx_update(
            approx2,
            precomputed.direct_nodes[node],
            precomputed.second_nodes2[node],
            precomputed.second_probs2[node],
        )

    selected1_set = set(normalized.campaign1)
    selected2_set = set(normalized.campaign2)
    common_nodes = selected1_set & selected2_set

    return SearchState(
        selected1=normalized.campaign1.copy(),
        selected2=normalized.campaign2.copy(),
        selected1_set=selected1_set,
        selected2_set=selected2_set,
        approx1=approx1,
        approx2=approx2,
        strength1=current_strength(precomputed.base_score1, initial_seeds.campaign1)
        + current_strength(precomputed.base_score1, normalized.campaign1),
        strength2=current_strength(precomputed.base_score2, initial_seeds.campaign2)
        + current_strength(precomputed.base_score2, normalized.campaign2),
        used_budget=normalized.total_size,
        action_index=normalized.total_size,
        count_s1_only=len(selected1_set - common_nodes),
        count_s2_only=len(selected2_set - common_nodes),
        count_both=len(common_nodes),
        approx_score=approximate_balanced_score(approx1, approx2),
        beam_score=0.0,
        lazy_cache={},
    )


def local_search_config(graph) -> dict[str, int | float]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        config: dict[str, int | float] = {
            "iterations": LOCAL_ITER_SMALL,
            "out_limit": LOCAL_OUT_LIMIT_SMALL,
            "in_limit": LOCAL_IN_LIMIT_SMALL,
            "approx_keep": LOCAL_APPROX_KEEP_SMALL,
            "eval_keep": LOCAL_EVAL_KEEP_SMALL,
            "quick_worlds": LOCAL_QUICK_WORLDS_SMALL,
            "tabu_size": LOCAL_TABU_SIZE_SMALL,
        }
    else:
        config = {
            "iterations": LOCAL_ITER_LARGE,
            "out_limit": LOCAL_OUT_LIMIT_LARGE,
            "in_limit": LOCAL_IN_LIMIT_LARGE,
            "approx_keep": LOCAL_APPROX_KEEP_LARGE,
            "eval_keep": LOCAL_EVAL_KEEP_LARGE,
            "quick_worlds": LOCAL_QUICK_WORLDS_LARGE,
            "tabu_size": LOCAL_TABU_SIZE_LARGE,
        }

    average_out_degree = graph.num_edges / max(1, graph.num_nodes)
    if graph.num_nodes > SMALL_GRAPH_THRESHOLD and average_out_degree >= LOCAL_DENSE_OUTDEG_THRESHOLD:
        config["iterations"] = min(int(config["iterations"]), 1)
        config["out_limit"] = min(int(config["out_limit"]), 1)
        config["in_limit"] = min(int(config["in_limit"]), 2)
        config["approx_keep"] = min(int(config["approx_keep"]), 4)
        config["eval_keep"] = min(int(config["eval_keep"]), 2)
        config["quick_worlds"] = min(int(config["quick_worlds"]), 3)
        config["tabu_size"] = min(int(config["tabu_size"]), 4)

    return config


def last_mile_config(graph) -> dict[str, int | float]:
    if graph.num_nodes <= SMALL_GRAPH_THRESHOLD:
        config: dict[str, int | float] = {
            "iterations": LAST_MILE_ITER_SMALL,
            "out_limit": LAST_MILE_OUT_LIMIT_SMALL,
            "in_limit": LAST_MILE_IN_LIMIT_SMALL,
            "approx_keep": LAST_MILE_APPROX_KEEP_SMALL,
            "eval_keep": LAST_MILE_EVAL_KEEP_SMALL,
            "quick_worlds": LAST_MILE_QUICK_WORLDS_SMALL,
            "tabu_size": LAST_MILE_TABU_SIZE_SMALL,
        }
    else:
        config = {
            "iterations": LAST_MILE_ITER_LARGE,
            "out_limit": LAST_MILE_OUT_LIMIT_LARGE,
            "in_limit": LAST_MILE_IN_LIMIT_LARGE,
            "approx_keep": LAST_MILE_APPROX_KEEP_LARGE,
            "eval_keep": LAST_MILE_EVAL_KEEP_LARGE,
            "quick_worlds": LAST_MILE_QUICK_WORLDS_LARGE,
            "tabu_size": LAST_MILE_TABU_SIZE_LARGE,
        }

    average_out_degree = graph.num_edges / max(1, graph.num_nodes)
    if graph.num_nodes > SMALL_GRAPH_THRESHOLD and average_out_degree >= LOCAL_DENSE_OUTDEG_THRESHOLD:
        config["iterations"] = min(int(config["iterations"]), 1)
        config["out_limit"] = min(int(config["out_limit"]), 1)
        config["in_limit"] = min(int(config["in_limit"]), 2)
        config["approx_keep"] = min(int(config["approx_keep"]), 4)
        config["eval_keep"] = min(int(config["eval_keep"]), 2)
        config["quick_worlds"] = min(int(config["quick_worlds"]), 2)
        config["tabu_size"] = min(int(config["tabu_size"]), 4)

    return config


def campaign_local_rank(
    precomputed: PrecomputedHeuristicData,
    campaign: int,
    node: int,
) -> float:
    if campaign == 1:
        return precomputed.coverage_strength1[node] + 0.3 * precomputed.common_strength[node]
    return precomputed.coverage_strength2[node] + 0.3 * precomputed.common_strength[node]


def common_local_rank(precomputed: PrecomputedHeuristicData, node: int) -> float:
    return precomputed.common_strength[node] + 0.4 * precomputed.combined_strength[node]


def build_seed_sets_from_sets(campaign1_nodes: set[int], campaign2_nodes: set[int]) -> SeedSets:
    return SeedSets(
        campaign1=sorted(campaign1_nodes),
        campaign2=sorted(campaign2_nodes),
    )


def register_local_neighbor(
    neighbors: dict[tuple[tuple[int, ...], tuple[int, ...]], SeedSets],
    campaign1_nodes: set[int],
    campaign2_nodes: set[int],
    budget: int,
) -> None:
    candidate = build_seed_sets_from_sets(campaign1_nodes, campaign2_nodes)
    if candidate.total_size > budget:
        return
    signature = solution_signature(candidate)
    neighbors.setdefault(signature, candidate)


def generate_local_neighbors(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    current_state: SearchState,
    current_solution: SeedSets,
    budget: int,
    config: dict[str, int | float],
) -> list[SeedSets]:
    current_campaign1 = set(current_solution.campaign1)
    current_campaign2 = set(current_solution.campaign2)
    only1 = list(current_campaign1 - current_campaign2)
    only2 = list(current_campaign2 - current_campaign1)
    common = list(current_campaign1 & current_campaign2)

    out_limit = int(config["out_limit"])
    in_limit = int(config["in_limit"])
    common_in_limit = max(1, min(in_limit, 2))

    weak_only1 = sorted(only1, key=lambda node: (campaign_local_rank(precomputed, 1, node), node))[:out_limit]
    weak_only2 = sorted(only2, key=lambda node: (campaign_local_rank(precomputed, 2, node), node))[:out_limit]
    weak_common = sorted(common, key=lambda node: (common_local_rank(precomputed, node), node))[:out_limit]
    mirror_from1 = sorted(only1, key=lambda node: (-precomputed.common_strength[node], node))[:out_limit]
    mirror_from2 = sorted(only2, key=lambda node: (-precomputed.common_strength[node], node))[:out_limit]

    initial_campaign1 = set(initial_seeds.campaign1)
    initial_campaign2 = set(initial_seeds.campaign2)
    pool1 = build_campaign_candidate_pool(
        graph,
        campaign=1,
        precomputed=precomputed,
        initial_campaign_nodes=initial_campaign1,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )[:in_limit]
    pool2 = build_campaign_candidate_pool(
        graph,
        campaign=2,
        precomputed=precomputed,
        initial_campaign_nodes=initial_campaign2,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )[:in_limit]
    common_pool = build_common_candidate_pool(
        graph,
        precomputed=precomputed,
        initial_campaign1=initial_campaign1,
        initial_campaign2=initial_campaign2,
        selected1=current_state.selected1_set,
        selected2=current_state.selected2_set,
        approx1=current_state.approx1,
        approx2=current_state.approx2,
    )[:common_in_limit]

    neighbors: dict[tuple[tuple[int, ...], tuple[int, ...]], SeedSets] = {}

    for removed in weak_only1:
        for added in pool1:
            register_local_neighbor(
                neighbors,
                (current_campaign1 - {removed}) | {added},
                current_campaign2,
                budget,
            )

    for removed in weak_only2:
        for added in pool2:
            register_local_neighbor(
                neighbors,
                current_campaign1,
                (current_campaign2 - {removed}) | {added},
                budget,
            )

    for removed in weak_common:
        for added in common_pool:
            register_local_neighbor(
                neighbors,
                (current_campaign1 - {removed}) | {added},
                (current_campaign2 - {removed}) | {added},
                budget,
            )

    for node in weak_only1:
        register_local_neighbor(
            neighbors,
            current_campaign1 - {node},
            current_campaign2 | {node},
            budget,
        )
    for node in weak_only2:
        register_local_neighbor(
            neighbors,
            current_campaign1 | {node},
            current_campaign2 - {node},
            budget,
        )

    for source in mirror_from1:
        for removed in weak_only2:
            register_local_neighbor(
                neighbors,
                current_campaign1,
                (current_campaign2 - {removed}) | {source},
                budget,
            )
    for source in mirror_from2:
        for removed in weak_only1:
            register_local_neighbor(
                neighbors,
                (current_campaign1 - {removed}) | {source},
                current_campaign2,
                budget,
            )

    pair_in_limit = max(1, min(in_limit, 2))
    for removed in weak_common[:1]:
        for added1 in pool1[:pair_in_limit]:
            for added2 in pool2[:pair_in_limit]:
                if added1 == added2:
                    continue
                register_local_neighbor(
                    neighbors,
                    (current_campaign1 - {removed}) | {added1},
                    (current_campaign2 - {removed}) | {added2},
                    budget,
                )

    for removed1 in weak_only1[: min(len(weak_only1), 2)]:
        for removed2 in weak_only2[: min(len(weak_only2), 2)]:
            for added in common_pool:
                register_local_neighbor(
                    neighbors,
                    (current_campaign1 - {removed1}) | {added},
                    (current_campaign2 - {removed2}) | {added},
                    budget,
                )

    current_signature = solution_signature(current_solution)
    neighbors.pop(current_signature, None)
    return list(neighbors.values())


def cached_approx_solution_score(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    balanced_seed_sets: SeedSets,
    approx_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
) -> float:
    signature = solution_signature(balanced_seed_sets)
    cached = approx_cache.get(signature)
    if cached is not None:
        return cached
    state = build_search_state_from_solution(graph, initial_seeds, precomputed, balanced_seed_sets)
    approx_cache[signature] = state.approx_score
    return state.approx_score


def cached_quick_solution_score(
    graph,
    initial_seeds: SeedSets,
    balanced_seed_sets: SeedSets,
    quick_worlds: int,
    eval_seed: int,
    quick_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float],
) -> float:
    signature = solution_signature(balanced_seed_sets)
    cached = quick_cache.get(signature)
    if cached is not None:
        return cached
    quick_score = estimate_solution_shared_worlds(
        graph=graph,
        initial_seeds=initial_seeds,
        balanced_seed_sets=balanced_seed_sets,
        num_worlds=quick_worlds,
        random_seed=eval_seed,
    )
    quick_cache[signature] = quick_score
    return quick_score


def refine_solution_with_config(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    budget: int,
    starting_solution: SeedSets,
    config: dict[str, int | float],
    min_seconds: float,
    timer: Timer | None = None,
) -> SeedSets:
    current_solution = normalize_seed_sets(starting_solution)
    current_state = build_search_state_from_solution(graph, initial_seeds, precomputed, current_solution)
    current_signature = solution_signature(current_solution)
    approx_cache = {current_signature: current_state.approx_score}
    quick_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    eval_seed = 20337000 + graph.num_nodes * 7 + budget * 131
    current_quick_score = cached_quick_solution_score(
        graph,
        initial_seeds,
        current_solution,
        int(config["quick_worlds"]),
        eval_seed,
        quick_cache,
    )
    best_solution = current_solution
    best_quick_score = current_quick_score
    tabu_signatures: deque[tuple[tuple[int, ...], tuple[int, ...]]] = deque(maxlen=int(config["tabu_size"]))

    for _ in range(int(config["iterations"])):
        if timer is not None and not timer.has_time(min_seconds):
            break
        neighbors = generate_local_neighbors(
            graph=graph,
            initial_seeds=initial_seeds,
            precomputed=precomputed,
            current_state=current_state,
            current_solution=current_solution,
            budget=budget,
            config=config,
        )
        if not neighbors:
            break

        approx_ranked: list[tuple[float, tuple[tuple[int, ...], tuple[int, ...]], SeedSets]] = []
        for candidate in neighbors:
            signature = solution_signature(candidate)
            if signature in tabu_signatures:
                continue
            approx_score = cached_approx_solution_score(
                graph,
                initial_seeds,
                precomputed,
                candidate,
                approx_cache,
            )
            approx_ranked.append((approx_score, signature, candidate))

        if not approx_ranked:
            break

        approx_ranked.sort(key=lambda item: (-item[0], item[1]))
        shortlisted = approx_ranked[: int(config["approx_keep"])]

        chosen_solution: SeedSets | None = None
        chosen_score = current_quick_score
        for _, _, candidate in shortlisted[: int(config["eval_keep"])]:
            if timer is not None and not timer.has_time(RERANK_MIN_SECONDS):
                break
            candidate_score = cached_quick_solution_score(
                graph,
                initial_seeds,
                candidate,
                int(config["quick_worlds"]),
                eval_seed,
                quick_cache,
            )
            if candidate_score > chosen_score + 1e-9:
                chosen_solution = candidate
                chosen_score = candidate_score

        if chosen_solution is None:
            break

        tabu_signatures.append(current_signature)
        current_solution = normalize_seed_sets(chosen_solution)
        current_signature = solution_signature(current_solution)
        current_state = build_search_state_from_solution(graph, initial_seeds, precomputed, current_solution)
        approx_cache[current_signature] = current_state.approx_score
        current_quick_score = chosen_score

        if current_quick_score > best_quick_score:
            best_solution = current_solution
            best_quick_score = current_quick_score

    return best_solution


def refine_solution_local_search(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    budget: int,
    starting_solution: SeedSets,
    timer: Timer | None = None,
) -> SeedSets:
    return refine_solution_with_config(
        graph=graph,
        initial_seeds=initial_seeds,
        precomputed=precomputed,
        budget=budget,
        starting_solution=starting_solution,
        config=local_search_config(graph),
        min_seconds=local_search_min_seconds(graph),
        timer=timer,
    )


def refine_solution_last_mile(
    graph,
    initial_seeds: SeedSets,
    precomputed: PrecomputedHeuristicData,
    budget: int,
    starting_solution: SeedSets,
    timer: Timer | None = None,
) -> SeedSets:
    return refine_solution_with_config(
        graph=graph,
        initial_seeds=initial_seeds,
        precomputed=precomputed,
        budget=budget,
        starting_solution=starting_solution,
        config=last_mile_config(graph),
        min_seconds=last_mile_min_seconds(graph),
        timer=timer,
    )


def build_heuristic_solution(graph, initial_seeds: SeedSets, budget: int) -> SeedSets:
    timer = create_heuristic_timer(graph)
    baseline_solution = build_timeout_baseline_solution(graph, initial_seeds, budget)
    best_solution = baseline_solution
    quick_worlds = PROFILE_EVAL_WORLDS_SMALL if graph.num_nodes <= SMALL_GRAPH_THRESHOLD else PROFILE_EVAL_WORLDS_LARGE
    best_score = estimate_solution_shared_worlds(
        graph=graph,
        initial_seeds=initial_seeds,
        balanced_seed_sets=baseline_solution,
        num_worlds=quick_worlds,
        random_seed=20271001 + budget * 31 + graph.num_nodes,
    )

    try:
        precomputed = build_precomputed_data(graph, initial_seeds, timer=timer)
    except TimeoutError:
        return baseline_solution

    candidate_profiles = select_top_profiles(graph, precomputed, initial_seeds, budget)

    if timer.expired():
        return baseline_solution

    for profile_index, profile in enumerate(candidate_profiles):
        if not timer.has_time(profile_min_seconds(graph)):
            break
        final_state = construct_solution_for_profile(
            graph=graph,
            initial_seeds=initial_seeds,
            precomputed=precomputed,
            profile=profile,
            budget=budget,
            timer=timer,
        )
        candidate_solution = SeedSets(
            campaign1=final_state.selected1.copy(),
            campaign2=final_state.selected2.copy(),
        )
        candidate_score = estimate_solution_shared_worlds(
            graph=graph,
            initial_seeds=initial_seeds,
            balanced_seed_sets=candidate_solution,
            num_worlds=quick_worlds,
            random_seed=20292000 + profile_index * 97 + profile_total_budget(profile),
        )
        if candidate_score > best_score:
            best_score = candidate_score
            best_solution = candidate_solution

    if timer.has_time(local_search_min_seconds(graph)):
        return refine_solution_local_search(
            graph,
            initial_seeds,
            precomputed,
            budget,
            best_solution,
            timer=timer,
        )
    if timer.has_time(last_mile_min_seconds(graph)):
        return refine_solution_last_mile(
            graph,
            initial_seeds,
            precomputed,
            budget,
            best_solution,
            timer=timer,
        )
    return best_solution


def build_submission_fallback_solution(graph, initial_seed_sets: SeedSets | None, budget: int) -> SeedSets:
    if graph is None or initial_seed_sets is None:
        return SeedSets(campaign1=[], campaign2=[])

    try:
        validate_seed_sets(initial_seed_sets, num_nodes=graph.num_nodes)
    except Exception:
        return SeedSets(campaign1=[], campaign2=[])

    for builder in (build_guard_baseline_solution, build_timeout_baseline_solution):
        try:
            candidate = builder(graph, initial_seed_sets, budget)
            validate_seed_sets(candidate, num_nodes=graph.num_nodes, budget=budget)
            return candidate
        except Exception:
            continue
    return SeedSets(campaign1=[], campaign2=[])


def main() -> None:
    args = parse_common_args(needs_output_path=False)
    graph = None
    initial_seed_sets: SeedSets | None = None
    try:
        graph = load_graph(args.network_path)
        initial_seed_sets = load_seed_sets(args.initial_seed_path)
        validate_seed_sets(initial_seed_sets, num_nodes=graph.num_nodes)
        try:
            balanced_seed_sets = build_heuristic_solution(graph, initial_seed_sets, args.budget)
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
