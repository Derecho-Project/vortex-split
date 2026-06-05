"""Unit tests for vortexsplit.core.export: extraction, glue coverage, run, save/load."""

import torch
import torch.fx as fx
import torch.nn as nn

from vortexsplit.core import export
from vortexsplit.core.models import SplitPlan, VortexGraph
from vortexsplit.core.trace import _boundaries, _dep_graph


class _Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4))


def _glue_tg() -> tuple[VortexGraph, fx.GraphModule]:
    """enc(relu, add w) -> proj(sin) -> mul-by-2 (UNOWNED glue) -> output.

    The mul node belongs to no submodule, exercising export's coverage pass.
    """
    root = _Root()
    g = fx.Graph()
    x = g.placeholder("x")
    w = g.get_attr("w")
    a = g.call_function(torch.relu, (x,))
    b = g.call_function(torch.add, (a, w))
    c = g.call_function(torch.sin, (b,))
    glue = g.call_function(torch.mul, (c, 2.0))  # unowned
    g.output(glue)
    gm = fx.GraphModule(root, g)

    def sub(uid, nodes):
        ins, outs = _boundaries(nodes)
        return VortexGraph.Submodule(uid, "X", nodes, ins, outs)

    submodules = {"enc": sub("enc", [a, w, b]), "proj": sub("proj", [c])}
    tg = VortexGraph(graph_module=gm, submodules=submodules, dep_graph=_dep_graph(submodules))
    return tg, gm


def _identity_plan(tg: VortexGraph) -> SplitPlan:
    return SplitPlan(partitions={uid: frozenset({uid}) for uid in tg.submodules}, objective_value=0.0)


def test_total_cover_absorbs_glue():
    tg, _ = _glue_tg()
    cover = export._total_cover(tg, _identity_plan(tg))
    all_covered = {n for ns in cover.values() for n in ns}
    compute = {n for n in tg.graph_module.graph.nodes if n.op not in ("placeholder", "output")}
    assert all_covered == compute, "every compute node (incl. glue) must be owned"
    # the mul glue lands in proj (forward pass from its producer c).
    proj_names = {n.name for n in cover["proj"]}
    assert "mul" in proj_names


def test_export_run_matches_monolith():
    torch.manual_seed(0)
    tg, gm = _glue_tg()
    x = torch.randn(4)
    reference = gm(x)

    artifact = export.export(tg, _identity_plan(tg))
    (out,) = artifact.run(x=x)
    assert torch.equal(out, reference), (reference, out)


def test_export_topology_and_specs():
    tg, _ = _glue_tg()
    artifact = export.export(tg, _identity_plan(tg))
    m = artifact.manifest
    assert m.graph_input_names == ("x",)
    assert m.output_names == ("mul",)
    assert m.topo_order == ("enc", "proj")  # enc produces the add, proj consumes it
    # enc emits the add node; proj consumes it and emits mul.
    assert m.producer_of["add"] == "enc"
    assert m.producer_of["mul"] == "proj"


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(1)
    tg, gm = _glue_tg()
    x = torch.randn(4)
    reference = gm(x)

    artifact = export.export(tg, _identity_plan(tg))
    artifact.save(tmp_path / "split.tspart")
    reloaded = export.SplitArtifact.load(tmp_path / "split.tspart")

    (out,) = reloaded.run(x=x)
    assert torch.equal(out, reference)
