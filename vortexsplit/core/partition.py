"""Apply a SplitPlan to a VortexGraph, and validate the Vortex diamond topology.

``apply`` merges the leaf components a plan groups together into one component each,
recomputing boundaries and the dependency graph with the same helpers ``trace`` uses, so
the result is an ordinary ``VortexGraph`` that ``profile`` and ``draw`` consume unchanged.
``validate_diamond`` checks that a graph reproduces the PreFLMR diamond: text ∥ vision →
Step C → Step D, with both branches feeding the join.
"""

from __future__ import annotations

from collections import defaultdict

import torch.fx as fx

from vortexsplit.core.models import DiamondReport, SplitPlan, VortexGraph
from vortexsplit.core.profile import static_profile
from vortexsplit.core.trace import _boundaries, _dep_graph


def apply(tg: VortexGraph, plan: SplitPlan) -> VortexGraph:
    """Merge ``tg``'s leaf components per ``plan`` into a new partitioned graph.

    Each plan partition becomes one ``Submodule`` whose interior is the union of its
    members' operators (restored to graph order), with boundaries and ``dep_graph``
    recomputed via the ``trace`` helpers and the shared ``graph_module`` reused. Static
    cost fields are refilled via ``static_profile``; runtime fields are left empty for the
    caller to re-profile (a merged component's activation peak isn't the sum of its parts).
    """
    order = {n: i for i, n in enumerate(tg.graph_module.graph.nodes)}
    merged: dict[str, VortexGraph.Submodule] = {}
    for new_uid, member_uids in plan.partitions.items():
        interior: list[fx.Node] = [n for u in member_uids for n in tg.submodules[u].interior_nodes]
        interior.sort(key=order.__getitem__)
        input_nodes, output_nodes = _boundaries(interior)
        # a single-leaf partition keeps its original type; a true merge is labelled.
        if len(member_uids) == 1:
            module_type = tg.submodules[next(iter(member_uids))].module_type
        else:
            module_type = f"vortexsplit.Partition[{len(member_uids)} leaves]"
        merged[new_uid] = VortexGraph.Submodule(
            semantic_uid=new_uid,
            module_type=module_type,
            interior_nodes=interior,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        )

    new_tg = VortexGraph(
        graph_module=tg.graph_module,
        submodules=merged,
        dep_graph=_dep_graph(merged),
    )
    static_profile(new_tg)  # exact, no execution — refills params / static bytes
    return new_tg


# check hand-split match
_ROLE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("transformer_mapping_input_linear", "step_c"),
    ("transformer_mapping_network", "step_d"),
    ("transformer_mapping_output_linear", "step_d"),
    ("query_text_encoder_linear", "text"),
    ("query_text_encoder", "text"),
    ("query_vision_encoder", "vision"),
    ("query_vision_projection", "vision"),
)

_REQUIRED_DIAMOND_EDGES: frozenset[tuple[str, str]] = frozenset(
    {
        ("vision", "step_c"),
        ("step_c", "step_d"),
        ("text", "step_d"),
    }
)


def _role_of(uid: str) -> str:
    for prefix, role in _ROLE_PREFIXES:
        if uid == prefix or uid.startswith(prefix + ".") or uid.startswith(prefix):
            return role
    return "other"


def validate_diamond(tg: VortexGraph) -> DiamondReport:
    roles = {uid: _role_of(uid) for uid in tg.submodules}

    role_edges: set[tuple[str, str]] = set()
    for src, sinks in tg.dep_graph.items():
        for dst in sinks:
            a, b = roles[src], roles[dst]
            if a != b:
                role_edges.add((a, b))

    has_upstream: dict[str, set[str]] = defaultdict(set)
    for a, b in role_edges:
        has_upstream[b].add(a)

    missing = _REQUIRED_DIAMOND_EDGES - role_edges
    text_is_source = not has_upstream.get("text")
    vision_is_source = not has_upstream.get("vision")

    problems = []
    if missing:
        problems.append(f"missing role edges {sorted(missing)}")
    if not text_is_source:
        problems.append(f"text has upstream {sorted(has_upstream['text'])}")
    if not vision_is_source:
        problems.append(f"vision has upstream {sorted(has_upstream['vision'])}")

    return DiamondReport(
        roles=roles,
        role_edges=frozenset(role_edges),
        is_diamond=not problems,
        detail="; ".join(problems) if problems else "diamond topology confirmed",
    )
