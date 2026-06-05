"""Unit tests for vortexsplit.core.dominance on tiny hand-built fx graphs.

These run on the CPU with no model download. networkx provides an independent
immediate-dominator oracle to cross-check the result.
"""

import networkx as nx
import torch
import torch.fx as fx

from vortexsplit.core import dominance
from vortexsplit.core.dominance import ENTRY, EXIT


# --------------------------------------------------------------------------- #
# graph builders — each returns an fx.Graph with named call_function nodes
# --------------------------------------------------------------------------- #
def _g(build) -> fx.Graph:
    g = fx.Graph()
    build(g)
    return g


def chain_graph() -> fx.Graph:
    """x -> a -> b -> c -> out."""

    def build(g: fx.Graph):
        x = g.placeholder("x")
        a = g.call_function(torch.relu, (x,))
        b = g.call_function(torch.relu, (a,))
        c = g.call_function(torch.relu, (b,))
        g.output(c)

    return _g(build)


def diamond_graph() -> fx.Graph:
    """x -> a -> {b, c} -> d -> out  (the canonical SESE diamond)."""

    def build(g: fx.Graph):
        x = g.placeholder("x")
        a = g.call_function(torch.relu, (x,))
        b = g.call_function(torch.sin, (a,))
        c = g.call_function(torch.cos, (a,))
        d = g.call_function(torch.add, (b, c))
        g.output(d)

    return _g(build)


def nested_diamond_graph() -> fx.Graph:
    """Outer diamond whose left arm is itself a diamond."""

    def build(g: fx.Graph):
        x = g.placeholder("x")
        a = g.call_function(torch.relu, (x,))
        # inner diamond on the left arm: a -> {l1, l2} -> lj
        l1 = g.call_function(torch.sin, (a,))
        l2 = g.call_function(torch.cos, (a,))
        lj = g.call_function(torch.add, (l1, l2))
        # right arm
        r = g.call_function(torch.neg, (a,))
        d = g.call_function(torch.add, (lj, r))
        g.output(d)

    return _g(build)


def multi_output_graph() -> fx.Graph:
    """Two returned tensors, exercising the tuple-aware EXIT augmentation."""

    def build(g: fx.Graph):
        x = g.placeholder("x")
        a = g.call_function(torch.relu, (x,))
        b = g.call_function(torch.sin, (a,))
        c = g.call_function(torch.cos, (a,))
        g.output((b, c))

    return _g(build)


def weighted_chain_graph() -> fx.Graph:
    """x -> lin1(w1) -> lin2(w2) -> out.

    Weights are get_attr roots; if they leaked into the CFG they would give lin2
    an alternate ENTRY path and break idom(lin2) == lin1. This locks that down.
    """

    def build(g: fx.Graph):
        x = g.placeholder("x")
        w1 = g.get_attr("w1")
        lin1 = g.call_function(torch.add, (x, w1))
        w2 = g.get_attr("w2")
        lin2 = g.call_function(torch.add, (lin1, w2))
        g.output(lin2)

    return _g(build)


# --------------------------------------------------------------------------- #
# networkx oracle
# --------------------------------------------------------------------------- #
def _nx_cfg(info: dominance.DominanceInfo) -> nx.DiGraph:
    """Reconstruct the analysed CFG as a networkx DiGraph for cross-checking."""
    dg = nx.DiGraph()
    for u, vs in info.succ.items():
        dg.add_node(u)
        for v in vs:
            dg.add_edge(u, v)
    return dg


def _check_idom_against_networkx(info: dominance.DominanceInfo):
    dg = _nx_cfg(info)
    oracle = dict(nx.immediate_dominators(dg, ENTRY))
    # nx maps the root to itself, same as us.
    ours = {n: info.idom[n] for n in oracle}
    assert ours == oracle, f"idom mismatch:\n ours={ours}\n nx  ={oracle}"


def _check_ipdom_against_networkx(info: dominance.DominanceInfo):
    # post-dominators = dominators on the reversed CFG rooted at EXIT.
    dg = _nx_cfg(info).reverse(copy=True)
    oracle = dict(nx.immediate_dominators(dg, EXIT))
    ours = {n: info.ipdom[n] for n in oracle}
    assert ours == oracle, f"ipdom mismatch:\n ours={ours}\n nx  ={oracle}"


ALL_GRAPHS = [
    chain_graph,
    diamond_graph,
    nested_diamond_graph,
    multi_output_graph,
    weighted_chain_graph,
]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _named(info: dominance.DominanceInfo) -> dict[str, fx.Node]:
    return {n.name: n for n in info.real_order}


def test_idom_matches_networkx_on_all_graphs():
    for builder in ALL_GRAPHS:
        info = dominance.analyze(builder())
        _check_idom_against_networkx(info)


def test_ipdom_matches_networkx_on_all_graphs():
    for builder in ALL_GRAPHS:
        info = dominance.analyze(builder())
        _check_ipdom_against_networkx(info)


def test_chain_dominator_tree():
    info = dominance.analyze(chain_graph())
    n = _named(info)  # fx names the three relu calls relu, relu_1, relu_2
    x, a, b, c = n["x"], n["relu"], n["relu_1"], n["relu_2"]
    # The placeholder is a real CFG node: ENTRY -> x -> a -> b -> c -> EXIT.
    assert info.idom[x] is ENTRY
    assert info.idom[a] is x
    assert info.idom[b] is a
    assert info.idom[c] is b
    assert info.dominates(a, c)
    assert not info.dominates(c, a)


def test_diamond_dominance_and_postdominance():
    info = dominance.analyze(diamond_graph())
    n = _named(info)
    a, b, c, d = n["relu"], n["sin"], n["cos"], n["add"]
    # a dominates the join d; d post-dominates a -> (a, d) is the SESE region.
    assert info.idom[d] is a, "join's idom is the fork, not a branch"
    assert info.dominates(a, d)
    assert info.postdominates(d, a)
    # neither branch dominates the join (parallel arms).
    assert not info.dominates(b, d)
    assert not info.dominates(c, d)
    assert info.dominates(ENTRY, d)


def test_weights_excluded_from_cfg_keep_chain_dominance():
    info = dominance.analyze(weighted_chain_graph())
    n = _named(info)
    lin1, lin2 = n["add"], n["add_1"]
    # The crux: lin1 must dominate lin2 despite w2 being an alternate root in the
    # raw dataflow. Weights are folded out, so the activation chain is preserved.
    assert info.idom[lin2] is lin1
    assert info.dominates(lin1, lin2)
    # get_attr nodes never enter the analysis node set.
    assert all(node.op != "get_attr" for node in info.real_order)


def test_multi_output_postdominators_well_defined():
    info = dominance.analyze(multi_output_graph())
    n = _named(info)
    a, b, c = n["relu"], n["sin"], n["cos"]
    # Both returns post-dominate nothing but EXIT; a is post-dominated by EXIT.
    assert info.postdominates(EXIT, a)
    assert info.postdominates(EXIT, b)
    # a does not post-dominate the two parallel outputs' join (there is none).
    assert info.ipdom[b] is EXIT
    assert info.ipdom[c] is EXIT
    # a's immediate post-dominator is EXIT too (its two users reconverge only there).
    assert info.ipdom[a] is EXIT


def test_determinism():
    # Same graph analysed twice yields identical idom/ipdom keyed by node name.
    g = nested_diamond_graph()
    i1 = dominance.analyze(g)
    # Re-analyse the same graph object; node identities are stable.
    i2 = dominance.analyze(g)
    assert {
        k.name if isinstance(k, fx.Node) else k: (v.name if isinstance(v, fx.Node) else v) for k, v in i1.idom.items()
    } == {
        k.name if isinstance(k, fx.Node) else k: (v.name if isinstance(v, fx.Node) else v) for k, v in i2.idom.items()
    }
