"""Unit tests for vortexsplit.core.optimize.

The headline test certifies the DP's optimality by brute-forcing every cut of a
small region tree across thousands of cost assignments and confirming the DP always
matches the true min-bottleneck.
"""

import itertools

import torch
import torch.fx as fx
import torch.nn as nn

from vortexsplit.core import optimize
from vortexsplit.core.models import SplitPlan, VortexGraph
from vortexsplit.core.sese import Region
from vortexsplit.core.trace import _boundaries, _dep_graph


# --------------------------------------------------------------------------- #
# tiny region-tree builders (fx_nodes empty -> can_keep defaults to True)
# --------------------------------------------------------------------------- #
def leaf(uid: str) -> Region:
    return Region(uid=uid, leaf_uids=frozenset({uid}), fx_nodes=frozenset(), children=())


def group(uid: str, *children: Region) -> Region:
    leaves = frozenset().union(*(c.leaf_uids for c in children))
    return Region(uid=uid, leaf_uids=leaves, fx_nodes=frozenset(), children=tuple(children))


def balanced_tree() -> Region:
    """root -> {g1 -> {a, b}, g2 -> {c, d}} (7 regions)."""
    return group("root", group("g1", leaf("a"), leaf("b")), group("g2", leaf("c"), leaf("d")))


# --------------------------------------------------------------------------- #
# brute-force oracle: enumerate every cut, take the min over max-stage cost
# --------------------------------------------------------------------------- #
def all_cuts(region: Region) -> list[frozenset[str]]:
    """Every antichain cut of the subtree, each as the set of chosen region uids."""
    cuts = [frozenset({region.uid})]  # keep whole
    if region.children:
        for combo in itertools.product(*(all_cuts(c) for c in region.children)):
            cuts.append(frozenset().union(*combo))
    return cuts


def brute_min_bottleneck(region: Region, cost: dict[str, float]) -> float:
    return min(max(cost[uid] for uid in cut) for cut in all_cuts(region))


def brute_min_bottleneck_capped(region: Region, cost: dict[str, float], cap: int) -> float:
    """Best bottleneck over cuts using at most ``cap`` partitions (∞ if none fit)."""
    feasible = [cut for cut in all_cuts(region) if len(cut) <= cap]
    if not feasible:
        return float("inf")
    return min(max(cost[uid] for uid in cut) for cut in feasible)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_dp_matches_brute_force_exhaustively():
    tree = balanced_tree()
    uids = ["root", "g1", "g2", "a", "b", "c", "d"]
    valid_cuts = set(map(frozenset, all_cuts(tree)))
    # exhaustively sweep every cost in {1,2,3} over all 7 regions: 3**7 = 2187 cases.
    for values in itertools.product((1.0, 2.0, 3.0), repeat=len(uids)):
        cost = dict(zip(uids, values))
        plan = optimize.solve_tree(tree, lambda r: cost[r.uid])
        # 1. optimal value matches brute force
        assert plan.objective_value == brute_min_bottleneck(tree, cost)
        # 2. the chosen partitions form a real cut
        assert frozenset(plan.partitions) in valid_cuts
        # 3. the cut covers every leaf exactly once
        covered = [u for members in plan.partitions.values() for u in members]
        assert sorted(covered) == ["a", "b", "c", "d"]


def test_capped_dp_matches_brute_force_exhaustively():
    tree = balanced_tree()
    uids = ["root", "g1", "g2", "a", "b", "c", "d"]
    for cap in (1, 2, 3, 4):
        valid = {frozenset(cut) for cut in all_cuts(tree) if len(cut) <= cap}
        # sweep costs in {1,2,3} over all 7 regions for each cap.
        for values in itertools.product((1.0, 2.0, 3.0), repeat=len(uids)):
            cost = dict(zip(uids, values))
            plan = optimize.solve_tree(tree, lambda r: cost[r.uid], max_partitions=cap)
            assert len(plan.partitions) <= cap
            assert plan.objective_value == brute_min_bottleneck_capped(tree, cost, cap)
            assert frozenset(plan.partitions) in valid


def test_cap_of_one_keeps_whole():
    tree = balanced_tree()
    cost = {"root": 7.0, "g1": 1.0, "g2": 1.0, "a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    plan = optimize.solve_tree(tree, lambda r: cost[r.uid], max_partitions=1)
    assert set(plan.partitions) == {"root"} and plan.objective_value == 7.0


def test_cap_limits_splitting_even_when_finer_is_cheaper():
    tree = balanced_tree()
    # unbounded would split to all 4 leaves (bottleneck 1); cap=2 forces g1|g2.
    cost = {"root": 100.0, "g1": 5.0, "g2": 5.0, "a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    plan = optimize.solve_tree(tree, lambda r: cost[r.uid], max_partitions=2)
    assert set(plan.partitions) == {"g1", "g2"}
    assert plan.objective_value == 5.0
    # with budget 4 it can reach the leaves.
    plan4 = optimize.solve_tree(tree, lambda r: cost[r.uid], max_partitions=4)
    assert set(plan4.partitions) == {"a", "b", "c", "d"}


def test_keep_whole_when_splitting_does_not_help():
    tree = balanced_tree()
    # the root as one stage is cheaper than any split's worst stage.
    cost = {"root": 1.0, "g1": 5.0, "g2": 5.0, "a": 9.0, "b": 9.0, "c": 9.0, "d": 9.0}
    plan = optimize.solve_tree(tree, lambda r: cost[r.uid])
    assert set(plan.partitions) == {"root"}
    assert plan.objective_value == 1.0


def test_split_when_a_subtree_is_a_bottleneck():
    tree = balanced_tree()
    # whole root is huge; g1/g2 each cheaper; leaves cheapest -> split fully.
    cost = {"root": 100.0, "g1": 50.0, "g2": 50.0, "a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    plan = optimize.solve_tree(tree, lambda r: cost[r.uid])
    assert set(plan.partitions) == {"a", "b", "c", "d"}
    assert plan.objective_value == 1.0


def test_ties_favour_coarser_partition():
    tree = balanced_tree()
    # keeping g1 whole ties with splitting it -> coarser (g1) wins.
    cost = {"root": 100.0, "g1": 2.0, "g2": 2.0, "a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0}
    plan = optimize.solve_tree(tree, lambda r: cost[r.uid])
    assert set(plan.partitions) == {"g1", "g2"}


# --------------------------------------------------------------------------- #
# collapse_by_modality (second pass over a DP plan)
# --------------------------------------------------------------------------- #
def _flmr_shaped_tg() -> VortexGraph:
    """A 7-unit FLMR-shaped graph: text & vision branches feeding a fusion."""
    g = fx.Graph()
    input_ids = g.placeholder("input_ids")
    pixel = g.placeholder("pixel_values")
    te = g.call_function(torch.relu, (input_ids,))  # query_text_encoder
    tl = g.call_function(torch.sin, (te,))  # query_text_encoder_linear
    ve = g.call_function(torch.relu, (pixel,))  # query_vision_encoder
    vp = g.call_function(torch.sin, (ve,))  # query_vision_projection
    c = g.call_function(torch.cos, (ve,))  # transformer_mapping_input_linear
    d = g.call_function(torch.add, (tl, c))  # transformer_mapping_network (multimodal)
    ol = g.call_function(torch.add, (d, vp))  # transformer_mapping_output_linear (multimodal)
    g.output(ol)
    gm = fx.GraphModule(nn.Module(), g)

    assign = {
        "query_text_encoder": [te],
        "query_text_encoder_linear": [tl],
        "query_vision_encoder": [ve],
        "query_vision_projection": [vp],
        "transformer_mapping_input_linear": [c],
        "transformer_mapping_network": [d],
        "transformer_mapping_output_linear": [ol],
    }
    subs = {}
    for uid, nodes in assign.items():
        ins, outs = _boundaries(nodes)
        subs[uid] = VortexGraph.Submodule(uid, "X", nodes, ins, outs)
    return VortexGraph(graph_module=gm, submodules=subs, dep_graph=_dep_graph(subs))


def _identity_plan(tg: VortexGraph) -> SplitPlan:
    return SplitPlan(partitions={u: frozenset({u}) for u in tg.submodules}, objective_value=0.0)


def test_collapse_by_modality_reproduces_diamond():
    tg = _flmr_shaped_tg()
    plan = optimize.collapse_by_modality(tg, _identity_plan(tg))
    groups = {frozenset(v) for v in plan.partitions.values()}
    assert groups == {
        frozenset({"query_text_encoder", "query_text_encoder_linear"}),  # A
        frozenset({"query_vision_encoder", "query_vision_projection"}),  # B
        frozenset({"transformer_mapping_input_linear"}),  # C
        frozenset({"transformer_mapping_network", "transformer_mapping_output_linear"}),  # D
    }
    # every leaf is covered exactly once.
    covered = sorted(u for members in plan.partitions.values() for u in members)
    assert covered == sorted(tg.submodules)


def test_collapse_without_family_refinement_gives_three_modalities():
    tg = _flmr_shaped_tg()
    plan = optimize.collapse_by_modality(tg, _identity_plan(tg), refine_by_family=False)
    groups = {frozenset(v) for v in plan.partitions.values()}
    # text, vision (incl. the bridge), and the multimodal fusion.
    assert groups == {
        frozenset({"query_text_encoder", "query_text_encoder_linear"}),
        frozenset({"query_vision_encoder", "query_vision_projection", "transformer_mapping_input_linear"}),
        frozenset({"transformer_mapping_network", "transformer_mapping_output_linear"}),
    }


def test_collapse_names_stage_by_primary_member():
    tg = _flmr_shaped_tg()
    plan = optimize.collapse_by_modality(tg, _identity_plan(tg))
    # the merged stage keeps the recognisable core module's uid (not a truncated prefix).
    assert "query_text_encoder" in plan.partitions  # {TE, TL} -> the encoder
    assert "query_vision_encoder" in plan.partitions  # {VE, VP} -> the encoder
    assert "transformer_mapping_network" in plan.partitions  # {D, OL} -> the network
    assert "transformer_mapping_input_linear" in plan.partitions  # singleton C keeps its name


def test_unkeepable_root_raises():
    # a root leaf that cannot be kept (and has no children to absorb it) is a hard
    # error: there is no convex covering to export.
    bad = leaf("x")
    try:
        optimize.solve_tree(bad, lambda r: 1.0, can_keep=lambda r: False)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "cannot be exported" in str(e)


def test_nonconvex_self_leaf_keeps_parent_whole():
    # 'enc' (a grouping) holds a non-convex self-leaf 'enc' plus a child 'enc.sub'.
    # The self-leaf can't be kept, so the optimizer keeps the whole 'enc' subtree.
    enc = group("enc", leaf("enc.sub"))
    enc_self = Region(uid="enc", leaf_uids=frozenset({"enc"}), fx_nodes=frozenset(), children=())
    enc = Region(
        uid="enc", leaf_uids=frozenset({"enc", "enc.sub"}), fx_nodes=frozenset(), children=(*enc.children, enc_self)
    )
    cost = {"enc": 10.0, "enc.sub": 1.0}
    plan = optimize.solve_tree(
        enc,
        lambda r: cost.get(r.uid, 0.0),
        can_keep=lambda r: r.leaf_uids != frozenset({"enc"}),  # self-leaf 'enc' not convex
    )
    assert set(plan.partitions) == {"enc"} and plan.partitions["enc"] == frozenset({"enc", "enc.sub"})
