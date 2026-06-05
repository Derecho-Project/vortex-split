"""SESE (single-entry/single-exit) analysis and the module-region tree.

find_sese_regions / program_structure_tree — the rigorous, operator-level dominance
result: a region ``(h, t)`` is SESE iff ``h`` dominates ``t`` and ``t`` post-dominates
``h``, so it is convex and independently exportable. These nest into a Program Structure
Tree.

build_region_tree + ConvexityOracle — the FQN module-region tree the optimizer cuts, plus
a convexity test (a region is convex iff no external operator lies on a path between two of
its operators).
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

import torch.fx as fx

from vortexsplit.core import dominance
from vortexsplit.core.dominance import ENTRY, EXIT, DominanceInfo, Node
from vortexsplit.core.models import VortexGraph


@dataclass(frozen=True)
class SESERegion:
    """A single-entry/single-exit region bounded by ``entry`` and ``exit``."""

    entry: Node
    exit: Node
    members: frozenset[Node]


def find_sese_regions(info: DominanceInfo) -> list[SESERegion]:
    """Canonical SESE regions of the analysed graph, outermost first."""
    reals = list(info.ordered_real_nodes())
    candidates = [ENTRY, *reals]
    rpo = info.rpo

    regions: list[SESERegion] = []
    for h in candidates:
        # nearest post-dominated exit t (smallest rpo > rpo[h]) with h dom t.
        best_t: Node | None = None
        for t in (*reals, EXIT):
            if rpo[t] <= rpo[h]:
                continue
            if info.dominates(h, t) and info.postdominates(t, h):
                if best_t is None or rpo[t] < rpo[best_t]:
                    best_t = t
        if best_t is None:
            continue
        members = frozenset(x for x in candidates + [EXIT] if info.dominates(h, x) and info.postdominates(best_t, x))
        interior = members - {h, best_t}
        if interior:  # skip trivial single-edge regions
            regions.append(SESERegion(entry=h, exit=best_t, members=members))
    return sorted(regions, key=lambda r: rpo[r.entry])


def program_structure_tree(info: DominanceInfo) -> dict[SESERegion, list[SESERegion]]:
    """Nest the canonical SESE regions into a (laminar) tree by member containment."""
    regions = find_sese_regions(info)
    children: dict[SESERegion, list[SESERegion]] = {r: [] for r in regions}
    for r in regions:
        parent = None
        for other in regions:
            if other is r or not r.members < other.members:
                continue
            if parent is None or other.members < parent.members:
                parent = other
        if parent is not None:
            children[parent].append(r)
    return children


class ConvexityOracle:
    """Answers "is this operator set independently exportable?" via reachability.

    ``universe`` makes the test glue-aware: only operators in it (e.g. the leaf-owned
    ones) count as occupancy, so unowned root-scope glue between ``S``'s operators does not
    disqualify a region — export absorbs that glue anyway, so the whole model is always
    keepable. ``universe=None`` is strict (every operator counts).
    """

    def __init__(self, graph: fx.Graph | fx.GraphModule, universe: frozenset[fx.Node] | None = None):
        if isinstance(graph, fx.GraphModule):
            graph = graph.graph
        self._succ: dict[fx.Node, tuple[fx.Node, ...]] = {}
        self._pred: dict[fx.Node, tuple[fx.Node, ...]] = {}
        for n in graph.nodes:
            self._succ[n] = tuple(n.users)
            self._pred[n] = tuple(n.all_input_nodes)
        self._universe = universe
        self._memo: dict[frozenset[fx.Node], bool] = {}

    def _reach(self, seeds: Iterable[fx.Node], adj: dict[fx.Node, tuple[fx.Node, ...]]) -> set[fx.Node]:
        seen: set[fx.Node] = set()
        queue = deque(seeds)
        while queue:
            n = queue.popleft()
            if n in seen:
                continue
            seen.add(n)
            queue.extend(adj.get(n, ()))
        return seen

    def is_convex(self, nodes: frozenset[fx.Node]) -> bool:
        if not nodes:
            return True
        if nodes in self._memo:
            return self._memo[nodes]
        down = self._reach(nodes, self._succ)
        up = self._reach(nodes, self._pred)
        between = (down & up) - nodes
        if self._universe is not None:
            between &= self._universe  # only other components count; glue is absorbed
        result = not between
        self._memo[nodes] = result
        return result


@dataclass(frozen=True)
class Region:
    """A candidate exportable component: a module subtree of the FQN hierarchy."""

    uid: str
    """dotted FQN prefix"""

    leaf_uids: frozenset[str]
    """subsumed leaf components"""

    fx_nodes: frozenset[fx.Node]
    """union of the leaves' interior operators"""

    children: tuple[Region, ...]
    """next-level FQN sub-regions"""

    def is_leaf_region(self) -> bool:
        return not self.children


def build_region_tree(tg: VortexGraph) -> Region:
    """Build the FQN module-region tree over ``tg``'s leaf components.

    Every dotted prefix of every leaf uid becomes a tree node covering all leaves at or
    below it. Pure-prefix nodes (a grouping FQN with no component of its own, e.g.
    ``query_text_encoder`` when only ``query_text_encoder.bert_model`` exists) are kept as
    internal grouping nodes. A uid that is *both* a component and a grouping FQN (owns ops
    directly *and* contains children) gets a synthetic leaf child holding its own ops, so
    splitting the grouping never orphans them.
    """
    interior = {uid: frozenset(sub.interior_nodes) for uid, sub in tg.submodules.items()}
    comps_of = {uid: tuple(uid.split(".")) for uid in tg.submodules}
    submodule_keys = {comps for comps in comps_of.values()}

    children_keys: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for comps in comps_of.values():
        for i in range(len(comps)):
            children_keys[comps[:i]].add(comps[: i + 1])

    def leaves_under(key: tuple[str, ...]) -> frozenset[str]:
        n = len(key)
        return frozenset(uid for uid, comps in comps_of.items() if comps[:n] == key)

    def make(key: tuple[str, ...]) -> Region:
        fqn_kids = [make(k) for k in sorted(children_keys.get(key, ()))]
        if key in submodule_keys and fqn_kids:
            own = ".".join(key)
            fqn_kids.append(Region(uid=own, leaf_uids=frozenset({own}), fx_nodes=interior[own], children=()))
        luids = leaves_under(key)
        fx_nodes = frozenset().union(*(interior[u] for u in luids)) if luids else frozenset()
        uid = ".".join(key) if key else "<ALL>"
        return Region(uid=uid, leaf_uids=luids, fx_nodes=fx_nodes, children=tuple(fqn_kids))

    return make(())


def iter_regions(root: Region) -> Iterable[Region]:
    """Pre-order walk over the region tree."""
    yield root
    for child in root.children:
        yield from iter_regions(child)


def analyze(tg: VortexGraph) -> tuple[Region, ConvexityOracle, DominanceInfo]:
    """Dominance + convexity oracle + module-region tree for ``tg``."""
    info = dominance.analyze(tg.graph_module)
    oracle = ConvexityOracle(tg.graph_module)
    tree = build_region_tree(tg)
    return tree, oracle, info
