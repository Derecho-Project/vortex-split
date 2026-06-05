"""Render a VortexGraph to a vector dataflow diagram via pydot/graphviz.

Lives outside models.py so the data containers stay import-light — the pydot
dependency (and the system `dot` binary) is only reached when someone actually
calls VortexGraph.draw().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch.fx as fx
import torch.nn as nn

if TYPE_CHECKING:
    from vortexsplit.core.models import VortexGraph


@dataclass
class _Box:
    """one rendered submodule node — a depth-collapsed group of submodules."""

    n_nodes: int = 0
    n_params: int = 0
    weight_bytes: int = 0
    act_bytes: int = 0
    module_type: str = ""
    n_members: int = 0


def _short(type_str: str) -> str:
    """'torch.nn.modules.linear.Linear' -> 'Linear'."""
    return type_str.rsplit(".", 1)[-1] if type_str else ""


def _human_bytes(n: int) -> str:
    """byte count -> a compact '1.5 MB'."""
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"  # unreachable — keeps the return non-Optional


def _resolve_attr(gm: fx.GraphModule, target: str):
    """A get_attr target -> its tensor (Parameter, buffer, or lifted constant).

    make_fx bakes weights in as get_attr nodes whose target is a dotted path
    into the GraphModule; resolve it the three ways torch itself would.
    """
    for getter in (gm.get_parameter, gm.get_buffer):
        try:
            return getter(target)
        except AttributeError:
            pass
    obj = gm
    try:
        for part in target.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj if hasattr(obj, "numel") else None


def _tensor_bytes(val) -> int:
    """numel*element_size summed over any tensors nested in val (one/tuple/dict)."""
    if hasattr(val, "numel") and hasattr(val, "element_size"):
        try:
            return val.numel() * val.element_size()
        except Exception:
            return 0
    if isinstance(val, (tuple, list)):
        return sum(_tensor_bytes(v) for v in val)
    if isinstance(val, dict):
        return sum(_tensor_bytes(v) for v in val.values())
    return 0


def _meta_bytes(tm) -> int:
    """Same, but off a ShapeProp TensorMetadata (shape + dtype) instead of a value."""
    if isinstance(tm, (tuple, list)):
        return sum(_meta_bytes(t) for t in tm)
    shape, dtype = getattr(tm, "shape", None), getattr(tm, "dtype", None)
    if shape is None or dtype is None:
        return 0
    numel = 1
    for d in shape:
        numel *= int(d)
    return numel * getattr(dtype, "itemsize", 0)


def _act_bytes(node: fx.Node) -> int:
    """Best-effort activation footprint of node's output, from trace metadata.

    Present when the graph carries example values ('val') or a ShapeProp
    'tensor_meta' — a trace with neither just yields 0, and render() then drops
    the activation line rather than printing a misleading zero.
    """
    if (val := node.meta.get("val")) is not None:
        return _tensor_bytes(val)
    return _meta_bytes(node.meta.get("tensor_meta"))


def _shape(node: fx.Node) -> str:
    """'[2×4 float32]' from a node's example value / tensor_meta, else ''."""
    val = node.meta.get("val")
    src = val if hasattr(val, "shape") else node.meta.get("tensor_meta")
    shape, dtype = getattr(src, "shape", None), getattr(src, "dtype", None)
    if shape is None or dtype is None:
        return ""
    dims = "×".join(str(int(d)) for d in shape)
    return f"[{dims} {str(dtype).removeprefix('torch.')}]"


def _collapse(uid: str, depth: int | None) -> str:
    """Truncate an FQN to its first `depth` components — absorbing anything
    deeper into its nearest depth-bounded ancestor. depth=None keeps it whole."""
    if depth is None:
        return uid
    parts = uid.split(".")
    return uid if len(parts) <= depth else ".".join(parts[:depth])


def render(tg: VortexGraph, path: Path | str, depth: int | None = None) -> Path:
    """Draw `tg` to `path`, format chosen by extension (.svg/.pdf). See VortexGraph.draw."""
    import pydot

    submodules, gm = tg.submodules, tg.graph_module

    # Collapse each submodule into the box keyed by its depth-bounded ancestor.
    members: dict[str, list[str]] = {}
    for uid in submodules:
        members.setdefault(_collapse(uid, depth), []).append(uid)

    # Aggregate each box, recording node -> box ownership. Weights are deduped (one
    # get_attr per weight); everything else contributes its activation bytes.
    node_box: dict[fx.Node, str] = {}
    boxes: dict[str, _Box] = {}
    for box, uids in members.items():
        bx = _Box(
            n_members=len(uids),
            module_type=_short(submodules[box].module_type) if box in submodules else "",
        )
        seen: set[str] = set()  # weight targets already tallied for this box
        for uid in uids:
            sub = submodules[uid]
            bx.n_nodes += len(sub.interior_nodes)
            for node in sub.interior_nodes:
                node_box[node] = box
                if node.op == "get_attr" and isinstance(node.target, str):
                    if node.target not in seen:
                        seen.add(node.target)
                        if isinstance(t := _resolve_attr(gm, node.target), nn.Parameter):
                            bx.n_params += t.numel()
                            bx.weight_bytes += t.numel() * t.element_size()
                else:
                    bx.act_bytes += _act_bytes(node)
        boxes[box] = bx
    show_act = any(bx.act_bytes for bx in boxes.values())

    # Self-standing nodes — graph I/O and anything no box claimed — each draw as their
    # own node, keyed apart from the FQN boxes so the id-spaces can't collide.
    standing = {n: f"<{n.op}:{n.name}>" for n in gm.graph.nodes if n not in node_box}
    key = {**node_box, **standing}  # every operator -> its diagram key

    # Edges off the fx graph: producer -> consumer between differing diagram nodes (the
    # box->box subset is dep_graph; the rest are the input->box / box->output links).
    edges: set[tuple[str, str]] = set()
    for node in gm.graph.nodes:
        dst = key[node]
        for producer in node.all_input_nodes:
            if (src := key[producer]) != dst:
                edges.add((src, dst))

    # rankdir TB so data reads top-to-bottom; components are filled boxes, graph I/O are
    # tinted ellipses, edges are the surviving dependencies.
    g = pydot.Dot("vortex", graph_type="digraph", rankdir="TB")
    g.set_node_defaults(
        shape="box",
        style="rounded,filled",
        fillcolor="#eef3fb",
        fontname="monospace",
        fontsize="10",
    )
    g.set_edge_defaults(color="#5577aa", arrowsize="0.7")

    ids = {k: f"n{i}" for i, k in enumerate(sorted(boxes))}
    for node in gm.graph.nodes:  # graph order keeps standing ids deterministic
        ids.setdefault(standing.get(node, ""), f"n{len(ids)}")
    ids.pop("", None)  # the placeholder default for owned (non-standing) nodes

    for box, bx in boxes.items():
        label = [box or "<ROOT>"]
        if bx.n_members > 1:
            label.append(f"▸ {bx.n_members} submodules")
        elif bx.module_type:
            label.append(bx.module_type)
        label.append(f"nodes: {bx.n_nodes}")
        label.append(f"params: {bx.n_params:,} ({_human_bytes(bx.weight_bytes)})")
        if show_act:
            label.append(f"act≈ {_human_bytes(bx.act_bytes)}")
        g.add_node(pydot.Node(ids[box], label="\n".join(label)))

    for node, k in standing.items():
        if node.op == "placeholder":  # graph input
            label = [f"in {node.name}", _shape(node)]
            fill = "#e7f6e7"
        elif node.op == "output":  # graph output
            label = ["output"]
            fill = "#fbeae7"
        else:
            label = [node.name]
            fill = "#eeeeee"
        text = "\n".join(line for line in label if line)
        g.add_node(pydot.Node(ids[k], label=text, shape="ellipse", fillcolor=fill))

    for src, dst in sorted(edges):
        g.add_edge(pydot.Edge(ids[src], ids[dst]))

    # .svg / .pdf go through `dot`; an unknown or missing suffix falls back to svg.
    out = Path(path)
    g.write(str(out), format=out.suffix.lstrip(".").lower() or "svg")
    return out
