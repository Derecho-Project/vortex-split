"""Unit tests for vortexsplit.core.sese on tiny hand-built graphs."""

import torch
import torch.fx as fx
import torch.nn as nn

from vortexsplit.core import dominance, sese
from vortexsplit.core.models import VortexGraph


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def diamond() -> fx.Graph:
    """x -> a -> {b, c} -> d -> out."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.relu, (x,))
    b = g.call_function(torch.sin, (a,))
    c = g.call_function(torch.cos, (a,))
    d = g.call_function(torch.add, (b, c))
    g.output(d)
    return g


def nested_diamond() -> fx.Graph:
    """Outer diamond a->{ (b->{b1,b2}->bj), c }->d, with a true inner SESE (b, bj)."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.relu, (x,))
    b = g.call_function(torch.sin, (a,))
    b1 = g.call_function(torch.neg, (b,))
    b2 = g.call_function(torch.abs, (b,))
    bj = g.call_function(torch.add, (b1, b2))
    c = g.call_function(torch.cos, (a,))
    d = g.call_function(torch.add, (bj, c))
    g.output(d)
    return g


def named(graph: fx.Graph) -> dict[str, fx.Node]:
    return {n.name: n for n in graph.nodes}


def gm(graph: fx.Graph) -> fx.GraphModule:
    return fx.GraphModule(nn.Module(), graph)


# --------------------------------------------------------------------------- #
# rigorous fx-level SESE
# --------------------------------------------------------------------------- #
def test_diamond_sese_region():
    info = dominance.analyze(diamond())
    n = named(diamond())  # rebuild for names; identities differ but names stable
    regions = sese.find_sese_regions(info)
    # exactly one non-trivial region: the fork->join (relu, add) holding {sin, cos}.
    assert len(regions) == 1
    r = regions[0]
    member_names = {m.name for m in r.members if isinstance(m, fx.Node)}
    assert r.entry.name == "relu"
    assert r.exit.name == "add"
    assert member_names == {"relu", "sin", "cos", "add"}
    assert n  # names available


def test_nested_diamond_pst():
    info = dominance.analyze(nested_diamond())
    regions = sese.find_sese_regions(info)
    by_exit = {r.exit.name: r for r in regions if isinstance(r.exit, fx.Node)}
    # inner region (sin, add) and outer region (relu, add_1).
    assert "add" in by_exit and "add_1" in by_exit
    inner, outer = by_exit["add"], by_exit["add_1"]
    assert {m.name for m in inner.members if isinstance(m, fx.Node)} == {"sin", "neg", "abs_1", "add"}
    assert inner.members < outer.members  # inner strictly nested in outer

    pst = sese.program_structure_tree(info)
    # outer is a parent of inner.
    assert inner in pst[outer]


# --------------------------------------------------------------------------- #
# convexity oracle
# --------------------------------------------------------------------------- #
def test_convexity_oracle():
    g = diamond()
    n = named(g)
    oracle = sese.ConvexityOracle(g)
    a, b, c, d = n["relu"], n["sin"], n["cos"], n["add"]
    # the full fork..join is convex; dropping a branch is not.
    assert oracle.is_convex(frozenset({a, b, c, d}))
    assert oracle.is_convex(frozenset({b, c}))  # parallel branches, nothing between
    assert not oracle.is_convex(frozenset({a, d}))  # b, c lie between a and d
    assert oracle.is_convex(frozenset({d}))  # singletons are trivially convex
    # memoisation returns a stable answer.
    assert oracle.is_convex(frozenset({a, d})) is False


def test_glue_aware_convexity_ignores_unowned_nodes():
    # a -> b -> c, where b is unowned "glue". The leaf set {a, c} is strictly
    # non-convex (b sits between), but glue-aware (universe = {a, c}) it is convex,
    # because export will absorb b into the partition.
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.relu, (x,))
    b = g.call_function(torch.sin, (a,))  # glue
    c = g.call_function(torch.cos, (b,))
    g.output(c)

    strict = sese.ConvexityOracle(g)
    assert not strict.is_convex(frozenset({a, c}))

    glue_aware = sese.ConvexityOracle(g, universe=frozenset({a, c}))
    assert glue_aware.is_convex(frozenset({a, c}))
    # the whole leaf universe is always convex under itself.
    assert glue_aware.is_convex(frozenset({a, c}))


# --------------------------------------------------------------------------- #
# module-region tree
# --------------------------------------------------------------------------- #
def _toy_tg() -> VortexGraph:
    """A graph whose nodes we partition into FQN-named leaf submodules:
    enc.a -> enc.b -> proj, mirroring a module hierarchy."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.relu, (x,))
    b = g.call_function(torch.sin, (a,))
    p = g.call_function(torch.cos, (b,))
    g.output(p)
    graph_module = gm(g)

    def sub(uid, nodes):
        return VortexGraph.Submodule(
            semantic_uid=uid,
            module_type="X",
            interior_nodes=nodes,
            input_nodes=[],
            output_nodes=[],
        )

    submodules = {
        "enc.a": sub("enc.a", [a]),
        "enc.b": sub("enc.b", [b]),
        "proj": sub("proj", [p]),
    }
    return VortexGraph(graph_module=graph_module, submodules=submodules, dep_graph={})


def test_build_region_tree():
    tg = _toy_tg()
    root = sese.build_region_tree(tg)
    assert root.uid == "<ALL>"
    assert root.leaf_uids == frozenset({"enc.a", "enc.b", "proj"})

    by_uid = {r.uid: r for r in sese.iter_regions(root)}
    # "enc" is a grouping node covering its two children; "proj" is a leaf region.
    assert "enc" in by_uid
    assert by_uid["enc"].leaf_uids == frozenset({"enc.a", "enc.b"})
    assert by_uid["enc.a"].is_leaf_region()
    assert by_uid["proj"].is_leaf_region()
    # fx_nodes of the grouping node are the union of its leaves' interior nodes.
    assert len(by_uid["enc"].fx_nodes) == 2


def _self_leaf_tg() -> VortexGraph:
    """'enc' owns ops directly AND contains 'enc.sub' — exercises the self-leaf."""
    g = fx.Graph()
    x = g.placeholder("x")
    a = g.call_function(torch.relu, (x,))  # enc's own op
    b = g.call_function(torch.sin, (a,))  # enc.sub
    c = g.call_function(torch.cos, (b,))  # enc's own op
    g.output(c)
    graph_module = gm(g)

    def sub(uid, nodes):
        return VortexGraph.Submodule(uid, "X", nodes, [], [])

    submodules = {"enc": sub("enc", [a, c]), "enc.sub": sub("enc.sub", [b])}
    return VortexGraph(graph_module=graph_module, submodules=submodules, dep_graph={})


def test_self_leaf_child_covers_own_ops():
    tg = _self_leaf_tg()
    root = sese.build_region_tree(tg)
    enc = next(r for r in sese.iter_regions(root) if r.uid == "enc" and r.children)
    # 'enc' as a grouping covers both leaves; splitting it yields a self-leaf child
    # ({'enc'}) and the 'enc.sub' child ({'enc.sub'}), together covering everything.
    assert enc.leaf_uids == frozenset({"enc", "enc.sub"})
    child_coverage = frozenset().union(*(c.leaf_uids for c in enc.children))
    assert child_coverage == enc.leaf_uids
    self_leaf = next(c for c in enc.children if c.leaf_uids == frozenset({"enc"}))
    assert self_leaf.is_leaf_region() and len(self_leaf.fx_nodes) == 2


def test_region_fx_nodes_are_convex_on_contiguous_modules():
    tg = _toy_tg()
    root = sese.build_region_tree(tg)
    oracle = sese.ConvexityOracle(tg.graph_module)
    for r in sese.iter_regions(root):
        if r.fx_nodes:
            assert oracle.is_convex(r.fx_nodes), f"{r.uid} not convex"
