import torch
import torch.fx as fx

from vortexsplit.core.models import VortexGraph


def _boundaries(interior: list[fx.Node]) -> tuple[list[fx.Node], list[fx.Node]]:
    """Input and output operators of an interior operator set, in topological order."""
    owned = set(interior)

    input_nodes, seen = [], set()
    for n in interior:
        for producer in n.all_input_nodes:
            if producer not in owned and producer not in seen:
                seen.add(producer)
                input_nodes.append(producer)

    output_nodes = [n for n in interior if any(u not in owned for u in n.users)]
    return input_nodes, output_nodes


def _dep_graph(submodules: dict[str, VortexGraph.Submodule]) -> dict[str, set[str]]:
    """Component-level dataflow dependency graph, keyed source -> sinks.

    A producer owned by one component that feeds another's interior is a direct edge.
    An unowned glue operator is transparent: recurse through its producers to the
    nearest owning component.
    """
    node_owner = {n: uid for uid, sub in submodules.items() for n in sub.interior_nodes}
    memo: dict[fx.Node, frozenset[str]] = {}

    def upstream_owners(node: fx.Node) -> frozenset[str]:
        """Component(s) reached walking back from ``node`` through unowned operators."""
        if (owner := node_owner.get(node)) is not None:
            return frozenset((owner,))
        if node not in memo:
            memo[node] = frozenset().union(*(upstream_owners(p) for p in node.all_input_nodes))
        return memo[node]

    deps: dict[str, set[str]] = {uid: set() for uid in submodules}
    for uid, sub in submodules.items():
        for producer in sub.input_nodes:
            for src in upstream_owners(producer):
                if src != uid:
                    deps[src].add(uid)
    return deps


def trace(
    fn,
    *args,
    tracing_mode="real",
    root_uid="<ROOT>",
    original_module: torch.nn.Module,
    **kwargs,
) -> VortexGraph:
    """Trace ``original_module``'s ``fn`` into a VortexGraph, calling fn(*args, **kwargs)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    fn._orig_mod = original_module

    graph_module = make_fx(fn, tracing_mode=tracing_mode, record_module_stack=True)(*args, **kwargs)

    # make_fx records every executed op; drop the unobserved ones.
    graph_module.graph.eliminate_dead_code()
    graph_module.recompile()

    if not any(n.meta.get("nn_module_stack") for n in graph_module.graph.nodes):
        raise RuntimeError(
            "trace: no node has 'nn_module_stack' — module provenance was not recorded. Check "
            "record_module_stack=True and that original_module owns the submodules fn invokes."
        )

    # Assign every operator to exactly one component, yielding a total, acyclic DAG.

    # Pass 1: tagged ops -> their innermost owning module's FQN ('' -> root).
    members: dict[str, list[fx.Node]] = {}
    module_type: dict[str, str] = {}
    for node in graph_module.graph.nodes:
        if stack := node.meta.get("nn_module_stack"):
            fqn, mod_cls = list(stack.values())[-1]
            uid = fqn or root_uid
            module_type[uid] = mod_cls
            members.setdefault(uid, []).append(node)

    root_cls = type(original_module)
    module_type.setdefault(root_uid, f"{root_cls.__module__}.{root_cls.__qualname__}")
    members.setdefault(root_uid, [])

    # Pass 2: lift untagged glue into the component owning its scope.
    owner: dict[fx.Node, str] = {n: uid for uid, ns in members.items() for n in ns}
    for node in graph_module.graph.nodes:
        if node in owner or node.meta.get("nn_module_stack"):
            continue
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "get_attr":
            uid = next((owner[u] for u in node.users if u in owner), root_uid)
        else:
            uid = root_uid

        if uid == root_uid:
            continue

        members[uid].append(node)
        owner[node] = uid

    # Restore topological order within each component after the appends.
    order = {n: i for i, n in enumerate(graph_module.graph.nodes)}
    for ns in members.values():
        ns.sort(key=order.__getitem__)

    submodules: dict[str, VortexGraph.Submodule] = {}
    for uid, interior in members.items():
        if not interior:  # empty root scope (no glue)
            continue
        input_nodes, output_nodes = _boundaries(interior)
        submodules[uid] = VortexGraph.Submodule(
            semantic_uid=uid,
            module_type=module_type[uid],
            interior_nodes=interior,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        )
    return VortexGraph(
        graph_module=graph_module,
        submodules=submodules,
        dep_graph=_dep_graph(submodules),
    )
