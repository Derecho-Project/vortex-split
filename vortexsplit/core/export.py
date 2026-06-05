"""Export a partitioned VortexGraph into standalone components + a runtime.

Each component becomes a self-contained ``fx.GraphModule`` (built with the same
``profile._extract_partition`` the profiler uses, so interior weights travel with it). The
components plus a :class:`~vortexsplit.core.models.Manifest` describing the inter-component
tensor routing form a :class:`SplitArtifact`, whose ``run`` executes the components in
dependency order — handling the diamond's fork/join, not just a linear chain.

Leaf components don't cover every operator: root-scope glue (the final concat /
``F.normalize``) is owned by no leaf. Before extraction ``_total_cover`` assigns every such
operator to a component by propagating ownership backward from its consumers (and forward
from its producers for output-feeding glue), so the components recompute the whole graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.fx as fx

from vortexsplit.core.models import Manifest, PartitionSpec, SplitPlan, VortexGraph
from vortexsplit.core.profile import _extract_partition
from vortexsplit.core.trace import _boundaries


def _total_cover(tg: VortexGraph, plan: SplitPlan) -> dict[str, list[fx.Node]]:
    """Assign every compute operator to exactly one component.

    Seeds ownership from the plan's leaf components, then attaches each unowned operator
    (root-scope glue, stray get_attr) backward to its earliest owned consumer's component,
    or — for operators feeding only the output — forward to a producer's. Returns
    ``uid -> interior operators`` in graph order.
    """
    order = {n: i for i, n in enumerate(tg.graph_module.graph.nodes)}
    owner: dict[fx.Node, str] = {}
    for uid, members in plan.partitions.items():
        for leaf in members:
            for n in tg.submodules[leaf].interior_nodes:
                owner[n] = uid

    nodes = list(tg.graph_module.graph.nodes)
    # Backward pass: glue inherits its earliest owned consumer's component (reverse graph
    # order visits consumers before producers, so users are already assigned).
    for n in reversed(nodes):
        if n.op in ("placeholder", "output") or n in owner:
            continue
        owned_users = [u for u in n.users if u in owner]
        if owned_users:
            owner[n] = owner[min(owned_users, key=order.__getitem__)]
    # Forward pass: anything left (feeds only the output) joins a producer's component.
    for n in nodes:
        if n.op in ("placeholder", "output") or n in owner:
            continue
        owned_prods = [p for p in n.all_input_nodes if p in owner]
        if owned_prods:
            owner[n] = owner[max(owned_prods, key=order.__getitem__)]
        else:  # fully isolated operator — its own singleton component
            owner[n] = f"<glue:{n.name}>"

    cover: dict[str, list[fx.Node]] = {uid: [] for uid in plan.partitions}
    for n, uid in owner.items():
        cover.setdefault(uid, []).append(n)
    for ns in cover.values():
        ns.sort(key=order.__getitem__)
    return cover


@dataclass
class SplitArtifact:
    """Exported components plus their wiring; ``run`` reproduces the monolith."""

    modules: dict[str, fx.GraphModule]
    manifest: Manifest

    def to(self, device: torch.device | str) -> SplitArtifact:
        self.modules = {uid: m.to(device) for uid, m in self.modules.items()}
        return self

    def eval(self) -> SplitArtifact:
        for m in self.modules.values():
            m.eval()
        return self

    def draw(self, path: Path | str) -> Path:
        """Render the inter-partition dataflow (the manifest) to an SVG/PDF."""
        return render_flow(self.manifest, path)

    @torch.no_grad()
    def run(self, **graph_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Execute the partitions in dependency order, routing tensors by node name.

        ``graph_inputs`` are keyed by the model-level placeholder names. Returns the
        graph's output tensors in their original return order.
        """
        missing = set(self.manifest.graph_input_names) - set(graph_inputs)
        if missing:
            raise ValueError(f"run() missing graph inputs {sorted(missing)}")

        data: dict[str, torch.Tensor] = {name: graph_inputs[name] for name in self.manifest.graph_input_names}
        specs = {s.uid: s for s in self.manifest.specs}
        for uid in self.manifest.topo_order:
            spec = specs[uid]
            args = tuple(data[name] for name in spec.input_node_names)
            outs = self.modules[uid](*args)  # _extract_partition always returns a tuple
            for name, tensor in zip(spec.output_node_names, outs):
                data[name] = tensor
        outputs = tuple(data[name] for name in self.manifest.output_names)
        return outputs

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "specs": [vars(s) for s in self.manifest.specs],
            "producer_of": dict(self.manifest.producer_of),
            "graph_input_names": list(self.manifest.graph_input_names),
            "output_names": list(self.manifest.output_names),
            "topo_order": list(self.manifest.topo_order),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for i, spec in enumerate(self.manifest.specs):
            torch.save(self.modules[spec.uid], out / f"{i}.pt")
        return out

    @staticmethod
    def load(path: Path | str) -> SplitArtifact:
        src = Path(path)
        manifest = load_manifest(src)
        modules = {spec.uid: torch.load(src / f"{i}.pt", weights_only=False) for i, spec in enumerate(manifest.specs)}
        return SplitArtifact(modules=modules, manifest=manifest)


def load_manifest(path: Path | str) -> Manifest:
    """Read just the wiring (manifest.json) from a saved split — no weights loaded."""
    raw = json.loads((Path(path) / "manifest.json").read_text())
    specs = tuple(
        PartitionSpec(s["uid"], tuple(s["input_node_names"]), tuple(s["output_node_names"])) for s in raw["specs"]
    )
    return Manifest(
        specs=specs,
        producer_of=raw["producer_of"],
        graph_input_names=tuple(raw["graph_input_names"]),
        output_names=tuple(raw["output_names"]),
        topo_order=tuple(raw["topo_order"]),
    )


def _topological_partition_order(
    uids: list[str],
    deps: dict[str, set[str]],
) -> tuple[str, ...]:
    """Kahn topological sort of partitions given producer->consumer edges (deps)."""
    indeg = {u: 0 for u in uids}
    for src, sinks in deps.items():
        for dst in sinks:
            indeg[dst] += 1
    ready = sorted(u for u in uids if indeg[u] == 0)
    order: list[str] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for dst in sorted(deps.get(u, ())):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort()
    if len(order) != len(uids):
        raise ValueError("partition dependency graph is cyclic — split is not exportable")
    return tuple(order)


def export(tg: VortexGraph, plan: SplitPlan) -> SplitArtifact:
    """Build a runnable :class:`SplitArtifact` for ``plan`` over ``tg``."""
    gm = tg.graph_module
    cover = _total_cover(tg, plan)

    # Build each partition's standalone GraphModule and its boundary node lists.
    specs: list[PartitionSpec] = []
    modules: dict[str, fx.GraphModule] = {}
    owner: dict[fx.Node, str] = {n: uid for uid, ns in cover.items() for n in ns}

    boundaries: dict[str, tuple[list[fx.Node], list[fx.Node]]] = {}
    for uid, interior in cover.items():
        ins, outs = _boundaries(interior)
        boundaries[uid] = (ins, outs)
        sub = VortexGraph.Submodule(uid, "partition", interior, ins, outs)
        modules[uid] = _extract_partition(gm, sub)
        specs.append(
            PartitionSpec(
                uid=uid,
                input_node_names=tuple(n.name for n in ins),
                output_node_names=tuple(n.name for n in outs),
            )
        )

    # producer_of: which partition emits each boundary node (its output set).
    producer_of: dict[str, str] = {}
    for uid, (_ins, outs) in boundaries.items():
        for n in outs:
            producer_of[n.name] = uid

    # partition dependency edges: producer-owner -> consumer for each external input.
    deps: dict[str, set[str]] = {uid: set() for uid in cover}
    for uid, (ins, _outs) in boundaries.items():
        for producer in ins:
            src = owner.get(producer)
            if src is not None and src != uid:
                deps[src].add(uid)

    topo = _topological_partition_order(list(cover), deps)

    graph_inputs = tuple(n.name for n in gm.graph.nodes if n.op == "placeholder")
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    output_names = tuple(n.name for n in output_node.all_input_nodes)

    manifest = Manifest(
        specs=tuple(specs),
        producer_of=producer_of,
        graph_input_names=graph_inputs,
        output_names=output_names,
        topo_order=topo,
    )
    return SplitArtifact(modules=modules, manifest=manifest).eval()


def render_flow(manifest: Manifest, path: Path | str) -> Path:
    """Draw the manifest's inter-partition dataflow: a box per stage, an edge per
    routed tensor (labelled with its node name), plus the graph inputs and output.

    Format is chosen by extension (.svg/.pdf); needs pydot + the system ``dot``.
    """
    import pydot

    g = pydot.Dot("flow", graph_type="digraph", rankdir="TB")
    g.set_node_defaults(shape="box", style="rounded,filled", fillcolor="#eef3fb", fontname="monospace", fontsize="11")
    g.set_edge_defaults(color="#5577aa", arrowsize="0.7", fontname="monospace", fontsize="8")

    # Safe graphviz ids (a ':' in an id is a node:port separator, '.' etc. also bite),
    # so map each logical node to an opaque id and carry the human name in the label.
    ids: dict[object, str] = {}

    def node_id(key: object) -> str:
        return ids.setdefault(key, f"n{len(ids)}")

    for spec in manifest.specs:
        g.add_node(pydot.Node(node_id(("stage", spec.uid)), label=spec.uid))
    for name in manifest.graph_input_names:
        g.add_node(pydot.Node(node_id(("in", name)), label=name, shape="ellipse", fillcolor="#e7f6e7"))
    output_id = node_id(("out",))
    g.add_node(pydot.Node(output_id, label="output", shape="ellipse", fillcolor="#fbeae7"))

    graph_inputs = set(manifest.graph_input_names)
    for spec in manifest.specs:
        for name in spec.input_node_names:
            producer = manifest.producer_of.get(name)
            if producer is not None:  # tensor from another stage
                g.add_edge(pydot.Edge(node_id(("stage", producer)), node_id(("stage", spec.uid)), label=name))
            elif name in graph_inputs:  # a model-level input (argument)
                g.add_edge(pydot.Edge(node_id(("in", name)), node_id(("stage", spec.uid)), label=name))
    for name in manifest.output_names:
        producer = manifest.producer_of.get(name)
        if producer is not None:
            g.add_edge(pydot.Edge(node_id(("stage", producer)), output_id, label=name))

    out = Path(path)
    g.write(str(out), format=out.suffix.lstrip(".").lower() or "svg")
    return out
