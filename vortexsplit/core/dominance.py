"""Dominance and post-dominance over the traced fx DAG, via Lengauer-Tarjan.

The ``make_fx`` graph is a pure DAG (control flow specialised away, dead code removed),
so it is reducible and dominators are well defined. We build a CFG over the activation
dataflow and run Lengauer-Tarjan (near-linear ``O(E·α(E,V))``) for immediate dominators;
running it on the reversed CFG rooted at EXIT gives immediate post-dominators. The result
is a :class:`DominanceInfo`, which :mod:`vortexsplit.core.sese` turns into SESE regions.

Weights (``get_attr`` operators) are dropped from the CFG: they have no producers, so
wiring ENTRY straight to each would add ``ENTRY -> weight -> consumer`` bypass paths and
flatten the dominator tree. They are treated as available everywhere; which component owns
a weight is decided later, at region snapping (matching trace.py's Pass 2).
"""

from __future__ import annotations

import enum
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import torch.fx as fx


class Virtual(enum.Enum):
    """The two synthetic CFG endpoints; hashable singletons that coexist with
    ``fx.Node`` keys in the dominance maps."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


# A CFG node is a real fx node or one of the two virtual endpoints.
Node = fx.Node | Virtual

ENTRY = Virtual.ENTRY
EXIT = Virtual.EXIT


def _is_real(node: fx.Node) -> bool:
    """An operator that participates in the activation CFG: not a foldable weight,
    not the output sentinel collapsed into EXIT."""
    return node.op not in ("get_attr", "output")


def _build_cfg(graph: fx.Graph) -> tuple[dict[Node, set[Node]], dict[Node, set[Node]], list[fx.Node]]:
    """Activation CFG of ``graph`` with virtual ENTRY/EXIT.

    Returns ``(succ, pred, real_nodes)``, adjacency including the endpoints. Edges follow
    activations only: weight producers are dropped, an operator with no activation
    producer is rooted at ENTRY, and every operator feeding ``output`` feeds EXIT.
    """
    real_nodes = [n for n in graph.nodes if _is_real(n)]
    real_set = set(real_nodes)

    succ: dict[Node, set[Node]] = defaultdict(set)
    pred: dict[Node, set[Node]] = defaultdict(set)

    def link(u: Node, v: Node) -> None:
        succ[u].add(v)
        pred[v].add(u)

    for n in real_nodes:
        activation_preds = [p for p in n.all_input_nodes if p in real_set]
        if activation_preds:
            for p in activation_preds:
                link(p, n)
        else:  # placeholder, or a node fed only by weights -> a CFG root
            link(ENTRY, n)

    outputs = [n for n in graph.nodes if n.op == "output"]
    if len(outputs) != 1:
        raise ValueError(f"expected exactly one output node, found {len(outputs)}")
    for producer in outputs[0].all_input_nodes:  # tuple-aware: flattens multi-tensor returns
        if producer in real_set:
            link(producer, EXIT)

    # Ensure both endpoints are present as keys even for a degenerate graph.
    succ.setdefault(EXIT, set())
    pred.setdefault(ENTRY, set())
    return succ, pred, real_nodes


def _prune_to_entry_exit_paths(
    succ: dict[Node, set[Node]],
    pred: dict[Node, set[Node]],
) -> set[Node]:
    """Operators on some ENTRY->EXIT path (reachable from ENTRY and able to reach EXIT).
    An unused placeholder dead-ends — it belongs to no region, so we drop it."""

    def reachable(roots: Iterable[Node], adj: dict[Node, set[Node]]) -> set[Node]:
        seen: set[Node] = set()
        queue = deque(roots)
        while queue:
            n = queue.popleft()
            if n in seen:
                continue
            seen.add(n)
            queue.extend(adj.get(n, ()))
        return seen

    return reachable([ENTRY], succ) & reachable([EXIT], pred)


def _lengauer_tarjan(
    root: Node,
    succ: Mapping[Node, Iterable[Node]],
    pred: Mapping[Node, Iterable[Node]],
) -> dict[Node, Node]:
    """Lengauer-Tarjan immediate dominators of the flowgraph rooted at ``root``.

    ``succ`` drives the depth-first spanning tree; ``pred`` supplies each vertex's
    predecessors for the semidominators. (Swapping the two and rooting at EXIT gives
    post-dominators.) Returns ``idom`` for every vertex reachable from ``root``. DFS and
    path compression are iterative so deep chains (thousands of operators) never hit the
    recursion limit.
    """
    # 1. Depth-first numbering. dfnum is 1-based; vertex[i] inverts it. The iterator-per-
    # frame stack emulates recursion so ``parent`` is a genuine DFS-tree edge.
    dfnum: dict[Node, int] = {}
    vertex: list[Node | None] = [None]
    parent: dict[Node, Node] = {}
    semi: dict[Node, int] = {}

    def discover(node: Node) -> None:
        dfnum[node] = len(vertex)
        vertex.append(node)
        semi[node] = dfnum[node]

    discover(root)
    work = [(root, iter(succ.get(root, ())))]
    while work:
        v, it = work[-1]
        for w in it:
            if w not in dfnum:
                parent[w] = v
                discover(w)
                work.append((w, iter(succ.get(w, ()))))
                break
        else:
            work.pop()
    n = len(vertex) - 1

    # 2. Link-eval forest with iterative path compression (label carries min semi).
    ancestor: dict[Node, Node] = {}
    label: dict[Node, Node] = {vertex[i]: vertex[i] for i in range(1, n + 1)}  # type: ignore[misc]

    def compress(v: Node) -> None:
        # Iterative path compression: collect the operators with a grandparent (those the
        # recursion would update), then update them deepest-first, as the recursion unwinds.
        path = []
        x = v
        while ancestor.get(ancestor.get(x)) is not None:  # type: ignore[arg-type]
            path.append(x)
            x = ancestor[x]
        for y in reversed(path):
            a = ancestor[y]
            if semi[label[a]] < semi[label[y]]:
                label[y] = label[a]
            ancestor[y] = ancestor[a]

    def evaluate(v: Node) -> Node:
        if ancestor.get(v) is None:
            return label[v]
        compress(v)
        return label[v]

    # 3. Compute semidominators (reverse DFS order) and tentative idoms via buckets.
    bucket: dict[Node, set[Node]] = defaultdict(set)
    idom: dict[Node, Node] = {}
    for i in range(n, 1, -1):
        w = vertex[i]
        for v in pred.get(w, ()):  # type: ignore[arg-type]
            if v not in dfnum:  # predecessor unreachable from root — ignore
                continue
            u = evaluate(v)
            if semi[u] < semi[w]:  # type: ignore[index]
                semi[w] = semi[u]  # type: ignore[index]
        bucket[vertex[semi[w]]].add(w)  # type: ignore[index, arg-type]
        pw = parent[w]  # type: ignore[index]
        ancestor[w] = pw  # type: ignore link(parent[w], w)
        for v in list(bucket[pw]):
            bucket[pw].discard(v)
            u = evaluate(v)
            idom[v] = u if semi[u] < semi[v] else pw

    # 4. Relativise tentative idoms to final immediate dominators (DFS order).
    for i in range(2, n + 1):
        w = vertex[i]
        if idom[w] is not vertex[semi[w]]:  # type: ignore[index]
            idom[w] = idom[idom[w]]  # type: ignore
    idom[root] = root
    return idom


@dataclass(frozen=True)
class DominanceInfo:
    """Dominator and post-dominator trees over a traced fx graph.

    ``idom[n]`` is ``n``'s immediate dominator; ``ipdom[n]`` its immediate
    post-dominator. Both trees are rooted at a self-loop (``idom[ENTRY] is ENTRY``,
    ``ipdom[EXIT] is EXIT``). ``rpo[n]`` is the reverse-postorder index used to
    order region discovery deterministically.
    """

    entry: Virtual
    exit: Virtual
    nodes: frozenset[Node]
    """Real fx nodes on an ENTRY->EXIT path, plus ENTRY and EXIT."""
    real_order: tuple[fx.Node, ...]
    """Participating fx nodes in topological (reverse-postorder) order."""
    rpo: dict[Node, int]
    idom: dict[Node, Node]
    ipdom: dict[Node, Node]
    dom_tree: dict[Node, frozenset[Node]]
    """Immediate-dominator children: ``dom_tree[n]`` = nodes whose idom is ``n``."""
    pdom_tree: dict[Node, frozenset[Node]]
    """Immediate-post-dominator children."""
    succ: dict[Node, frozenset[Node]]
    pred: dict[Node, frozenset[Node]]

    def dominates(self, a: Node, b: Node) -> bool:
        """True iff every ENTRY->b path passes through ``a`` (reflexive: a==b)."""
        x = b
        while True:
            if x is a:
                return True
            nxt = self.idom.get(x)
            if nxt is None or nxt is x:  # reached the root without meeting a
                return False
            x = nxt

    def postdominates(self, a: Node, b: Node) -> bool:
        """True iff every b->EXIT path passes through ``a`` (reflexive: a==b)."""
        x = b
        while True:
            if x is a:
                return True
            nxt = self.ipdom.get(x)
            if nxt is None or nxt is x:
                return False
            x = nxt

    def dom_children(self, node: Node) -> frozenset[Node]:
        """Nodes immediately dominated by ``node`` (its dominator-tree children)."""
        return self.dom_tree.get(node, frozenset())

    def pdom_children(self, node: Node) -> frozenset[Node]:
        """Nodes immediately post-dominated by ``node``."""
        return self.pdom_tree.get(node, frozenset())

    def ordered_real_nodes(self) -> Iterator[fx.Node]:
        """Participating fx nodes in RPO order (ENTRY/EXIT excluded)."""
        yield from self.real_order


def _invert_tree(parent: dict[Node, Node]) -> dict[Node, frozenset[Node]]:
    children: dict[Node, set[Node]] = defaultdict(set)
    for n, p in parent.items():
        if n is not p:  # skip the root's self-loop
            children[p].add(n)
    return {n: frozenset(children.get(n, ())) for n in parent}


def analyze(graph: fx.Graph | fx.GraphModule) -> DominanceInfo:
    """Compute dominators and post-dominators for a traced fx graph."""
    if isinstance(graph, fx.GraphModule):
        graph = graph.graph

    succ, pred, real_nodes = _build_cfg(graph)
    keep = _prune_to_entry_exit_paths(succ, pred) | {ENTRY, EXIT}

    # Order adjacency by graph order so the DFS tree is reproducible across runs (fx.Node
    # sets otherwise iterate in id()-hash order). Correctness is DFS-tree-independent.
    kept_real = [n for n in real_nodes if n in keep]
    rank = {ENTRY: -1, EXIT: len(kept_real), **{n: i for i, n in enumerate(kept_real)}}
    succ = {u: sorted((v for v in vs if v in keep), key=rank.__getitem__) for u, vs in succ.items() if u in keep}
    pred = {u: sorted((v for v in vs if v in keep), key=rank.__getitem__) for u, vs in pred.items() if u in keep}
    succ.setdefault(EXIT, [])
    pred.setdefault(ENTRY, [])

    # Dominators from the forward flowgraph (ENTRY, succ), post-dominators from the
    # reversed one (EXIT, pred). fwd_order numbers ENTRY..EXIT only to order region
    # discovery deterministically downstream.
    fwd_order: list[Node] = [ENTRY, *kept_real, EXIT]

    idom = _lengauer_tarjan(ENTRY, succ, pred)
    ipdom = _lengauer_tarjan(EXIT, pred, succ)

    return DominanceInfo(
        entry=ENTRY,
        exit=EXIT,
        nodes=frozenset(keep),
        real_order=tuple(kept_real),
        rpo={n: i for i, n in enumerate(fwd_order)},
        idom=idom,
        ipdom=ipdom,
        dom_tree=_invert_tree(idom),
        pdom_tree=_invert_tree(ipdom),
        succ={u: frozenset(vs) for u, vs in succ.items()},
        pred={u: frozenset(vs) for u, vs in pred.items()},
    )
