"""The split optimizer: a DP over the module-region tree.

Given the module-region tree (:mod:`vortexsplit.core.sese`) and a per-region stage cost,
choose the cut that minimizes the pipeline bottleneck — the slowest stage. A cut assigns
every leaf to exactly one chosen region. Classic DP problem.

    dp(region) = min(
        stage_cost(region),                    # keep the whole module subtree as one stage
        max(dp(child) for child in children),  # or split into the children's best cuts
    )

"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch.fx as fx

from vortexsplit.core.models import SplitPlan, VortexGraph
from vortexsplit.core.sese import ConvexityOracle, Region, build_region_tree


def solve_tree(
    root: Region,
    stage_cost: Callable[[Region], float],
    can_keep: Callable[[Region], bool] = lambda _r: True,
    *,
    max_partitions: int | None = None,
) -> SplitPlan:
    """Min-bottleneck cut of the region tree."""
    if max_partitions is None:
        value, chosen = _dp_unbounded(root, stage_cost, can_keep)
    else:
        value, chosen = _dp_capped(root, stage_cost, can_keep, max_partitions)
    if math.isinf(value):
        raise ValueError(f"region {root.uid!r} cannot be exported within the partition cap")
    return SplitPlan(partitions={r.uid: r.leaf_uids for r in chosen}, objective_value=value)


def _dp_unbounded(
    region: Region,
    stage_cost: Callable[[Region], float],
    can_keep: Callable[[Region], bool],
) -> tuple[float, list[Region]]:
    """Min-bottleneck with no partition limit: keep a region whole or split it."""
    options: list[tuple[float, list[Region]]] = []
    if region.leaf_uids and can_keep(region):
        options.append((stage_cost(region), [region]))  # keep whole — listed first
    if region.children:
        child = [_dp_unbounded(c, stage_cost, can_keep) for c in region.children]
        split_value = max(v for v, _ in child)
        split_choice = [r for _, chosen in child for r in chosen]
        options.append((split_value, split_choice))
    if not options:
        return math.inf, []  # dead-end region; an ancestor must absorb it
    return min(options, key=lambda o: o[0])  # min is stable -> ties keep whole


def _dp_capped(
    region: Region,
    stage_cost: Callable[[Region], float],
    can_keep: Callable[[Region], bool],
    cap: int,
) -> tuple[float, list[Region]]:
    """Lowest bottleneck for ``region`` using at most ``cap`` partitions."""
    table = _budget_table(region, stage_cost, can_keep, cap)
    if not table:
        return math.inf, []
    _budget, (value, chosen) = min(table.items(), key=lambda kv: (kv[1][0], kv[0]))
    return value, chosen


def _budget_table(
    region: Region,
    stage_cost: Callable[[Region], float],
    can_keep: Callable[[Region], bool],
    cap: int,
) -> dict[int, tuple[float, list[Region]]]:
    """``budget -> (bottleneck, chosen regions)`` for using *exactly* that many partitions
    on ``region``'s subtree, for every feasible budget in ``1..cap``."""
    best: dict[int, tuple[float, list[Region]]] = {}
    if region.leaf_uids and can_keep(region):
        best[1] = (stage_cost(region), [region])  # whole subtree as one stage

    if region.children:
        # Knapsack the budget across children, minimising the worst child bottleneck.
        # combined[b] = best (max-bottleneck, choice) over processed children using b stages.
        combined: dict[int, tuple[float, list[Region]]] = {0: (0.0, [])}
        for child in region.children:
            child_table = _budget_table(child, stage_cost, can_keep, cap)
            nxt: dict[int, tuple[float, list[Region]]] = {}
            for used, (worst, choice) in combined.items():
                for cb, (cval, cchoice) in child_table.items():
                    total = used + cb
                    if total > cap:
                        continue
                    value = max(worst, cval)
                    if total not in nxt or value < nxt[total][0]:
                        nxt[total] = (value, choice + cchoice)
            combined = nxt
            if not combined:
                break  # a child is infeasible -> the region cannot be split
        for b, (worst, choice) in combined.items():
            if b == 0:
                continue  # every child must contribute at least one stage
            if b not in best or worst < best[b][0]:
                best[b] = (worst, choice)
    return best


@dataclass(frozen=True)
class CostModel:
    """Turns a profiled ``VortexGraph`` into a per-region stage cost (in ns)."""

    tg: VortexGraph
    batch_size: int
    bandwidth_bytes_per_ns: float = 100.0  # ~100 GB/s; raise toward inf for compute-only

    def _leaf_compute_ns(self, uid: str) -> float:
        trials = self.tg.submodules[uid].model_runtime_ns.get(self.batch_size)
        return sum(trials) / len(trials) if trials else 0.0

    def compute_ns(self, region: Region) -> float:
        return sum(self._leaf_compute_ns(u) for u in region.leaf_uids)

    def transfer_ns(self, region: Region) -> float:
        from vortexsplit.core.draw import _act_bytes
        from vortexsplit.core.trace import _boundaries

        if not region.fx_nodes:
            return 0.0
        ins, outs = _boundaries(list(region.fx_nodes))
        boundary_bytes = sum(_act_bytes(n) for n in ins) + sum(_act_bytes(n) for n in outs)
        return boundary_bytes / self.bandwidth_bytes_per_ns

    def stage_cost(self, region: Region) -> float:
        return self.compute_ns(region) + self.transfer_ns(region)


def solve(
    tg: VortexGraph,
    *,
    batch_size: int,
    bandwidth_bytes_per_ns: float = 100.0,
    max_partitions: int | None = None,
) -> SplitPlan:
    """Optimize the split of a profiled ``VortexGraph`` for pipeline throughput.

    ``max_partitions`` caps the number of exported stages. Leave it ``None`` for the
    unbounded throughput-optimal split, which shatters cheap subtrees into tiny stages.
    """
    tree = build_region_tree(tg)
    # glue-aware: only leaf-owned operators count as occupancy, so a region isn't
    # disqualified by unowned root-scope glue (which export absorbs). The whole model stays
    # keepable, so no partition cap >= 1 is ever infeasible.
    universe = frozenset(n for sub in tg.submodules.values() for n in sub.interior_nodes)
    oracle = ConvexityOracle(tg.graph_module, universe=universe)
    cost = CostModel(tg, batch_size, bandwidth_bytes_per_ns)

    def can_keep(region: Region) -> bool:
        return bool(region.fx_nodes) and oracle.is_convex(region.fx_nodes)

    return solve_tree(tree, cost.stage_cost, can_keep, max_partitions=max_partitions)


def collapse_by_modality(tg: VortexGraph, plan: SplitPlan, *, refine_by_family: bool = True) -> SplitPlan:
    """Merge a plan's components into per-modality stages — a pure coarsening.

    Only merges that keep every stage convex are allowed; a non-convex group is a hard
    error. The input ``plan`` must be at least as fine as the target.

    Each modality group is further split by the module's first two name tokens if
    refine_by_family is True.
    """
    graph = tg.graph_module.graph

    # Transitive placeholder provenance per operator, in one topological pass.
    deps: dict[fx.Node, frozenset[str]] = {}
    for node in graph.nodes:
        if node.op == "placeholder":
            deps[node] = frozenset({node.name})
        elif node.op != "output":
            deps[node] = frozenset().union(*(deps.get(i, frozenset()) for i in node.all_input_nodes))

    part_nodes = {
        uid: frozenset(n for leaf in leaves for n in tg.submodules[leaf].interior_nodes)
        for uid, leaves in plan.partitions.items()
    }

    def signature(uid: str) -> frozenset[str]:
        return frozenset().union(*(deps.get(n, frozenset()) for n in part_nodes[uid]))

    def family(uid: str) -> str:
        return "_".join(uid.split(".")[0].split("_")[:2])

    def key(uid: str):
        return (signature(uid), family(uid)) if refine_by_family else (signature(uid),)

    groups: dict[object, list[str]] = defaultdict(list)
    for uid in plan.partitions:
        groups[key(uid)].append(uid)

    universe = frozenset(n for sub in tg.submodules.values() for n in sub.interior_nodes)
    oracle = ConvexityOracle(tg.graph_module, universe=universe)

    partitions: dict[str, frozenset[str]] = {}
    for uids in groups.values():
        nodes = frozenset(n for u in uids for n in part_nodes[u])
        if not oracle.is_convex(nodes):
            raise ValueError(f"modality group {sorted(uids)} is not convex — cannot merge safely")
        leaves = {u: plan.partitions[u] for u in uids}
        partitions[_group_name(leaves)] = frozenset(leaf for members in leaves.values() for leaf in members)

    return SplitPlan(
        partitions=partitions, objective_value=plan.objective_value, objective=plan.objective + "+modality"
    )


def _group_name(leaves: dict[str, frozenset[str]]) -> str:
    """Name a merged stage after its primary member — the one covering the most leaves (ties
    broken by shorter name)."""
    return min(leaves, key=lambda u: (-len(leaves[u]), len(u), u))


def expand_to_fx_nodes(plan: SplitPlan, tg: VortexGraph) -> dict[str, frozenset[fx.Node]]:
    """Resolve each component's leaf uids to the union of their interior operators."""
    return {
        uid: frozenset(n for leaf in members for n in tg.submodules[leaf].interior_nodes)
        for uid, members in plan.partitions.items()
    }
