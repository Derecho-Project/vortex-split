"""Unit tests for vortexsplit.core.partition: apply() and validate_diamond()."""

import torch
import torch.fx as fx
import torch.nn as nn

from vortexsplit.core import partition
from vortexsplit.core.models import SplitPlan, VortexGraph
from vortexsplit.core.trace import _boundaries, _dep_graph


def _make_tg(graph: fx.Graph, assignment: dict[str, list[fx.Node]]) -> VortexGraph:
    """Wrap a hand-built graph + leaf assignment into a VortexGraph with boundaries
    and a dep_graph computed by the real trace helpers."""
    gm = fx.GraphModule(nn.Module(), graph)
    subs: dict[str, VortexGraph.Submodule] = {}
    for uid, nodes in assignment.items():
        ins, outs = _boundaries(nodes)
        subs[uid] = VortexGraph.Submodule(uid, "X", nodes, ins, outs)
    return VortexGraph(graph_module=gm, submodules=subs, dep_graph=_dep_graph(subs))


def _diamond_tg() -> VortexGraph:
    """FLMR-shaped diamond with FQN-named leaves matching the role table."""
    g = fx.Graph()
    input_ids = g.placeholder("input_ids")
    pixel = g.placeholder("pixel_values")
    text_enc = g.call_function(torch.relu, (input_ids,))
    text_lin = g.call_function(torch.sin, (text_enc,))
    vis_enc = g.call_function(torch.relu, (pixel,))
    vis_proj = g.call_function(torch.sin, (vis_enc,))
    step_c = g.call_function(torch.cos, (vis_enc,))
    step_d = g.call_function(torch.add, (text_enc, step_c))
    out_lin = g.call_function(torch.add, (step_d, text_lin))
    final = g.call_function(torch.add, (out_lin, vis_proj))
    g.output(final)

    assignment = {
        "query_text_encoder.bert_model": [text_enc],
        "query_text_encoder_linear": [text_lin],
        "query_vision_encoder.vision_model": [vis_enc],
        "query_vision_projection.model": [vis_proj],
        "transformer_mapping_input_linear": [step_c],
        "transformer_mapping_network.layer": [step_d],
        "transformer_mapping_output_linear": [out_lin],
    }
    return _make_tg(g, assignment)


# --------------------------------------------------------------------------- #
# validate_diamond
# --------------------------------------------------------------------------- #
def test_validate_diamond_positive_on_leaf_graph():
    report = partition.validate_diamond(_diamond_tg())
    assert report.is_diamond, report.detail
    assert ("vision", "step_c") in report.role_edges
    assert ("step_c", "step_d") in report.role_edges
    assert ("text", "step_d") in report.role_edges


def test_validate_diamond_negative_when_edge_missing():
    # Rewire so Step C is fed by text, not vision -> required (vision, step_c) absent.
    g = fx.Graph()
    input_ids = g.placeholder("input_ids")
    pixel = g.placeholder("pixel_values")
    text_enc = g.call_function(torch.relu, (input_ids,))
    vis_enc = g.call_function(torch.relu, (pixel,))
    step_c = g.call_function(torch.cos, (text_enc,))  # wrong source
    step_d = g.call_function(torch.add, (text_enc, step_c))
    out_lin = g.call_function(torch.add, (step_d, vis_enc))
    g.output(out_lin)
    tg = _make_tg(
        g,
        {
            "query_text_encoder.bert_model": [text_enc],
            "query_vision_encoder.vision_model": [vis_enc],
            "transformer_mapping_input_linear": [step_c],
            "transformer_mapping_network.layer": [step_d],
            "transformer_mapping_output_linear": [out_lin],
        },
    )
    report = partition.validate_diamond(tg)
    assert not report.is_diamond
    assert "vision" in report.detail and "step_c" in report.detail


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def test_apply_merges_and_preserves_diamond():
    tg = _diamond_tg()
    # merge the text encoder with its linear into one partition; keep the rest.
    plan = SplitPlan(
        partitions={
            "query_text_encoder": frozenset({"query_text_encoder.bert_model", "query_text_encoder_linear"}),
            "query_vision_encoder.vision_model": frozenset({"query_vision_encoder.vision_model"}),
            "query_vision_projection.model": frozenset({"query_vision_projection.model"}),
            "transformer_mapping_input_linear": frozenset({"transformer_mapping_input_linear"}),
            "transformer_mapping_network.layer": frozenset({"transformer_mapping_network.layer"}),
            "transformer_mapping_output_linear": frozenset({"transformer_mapping_output_linear"}),
        },
        objective_value=0.0,
    )
    merged = partition.apply(tg, plan)

    # the merged partition holds both leaves' interior nodes, in graph order.
    text = merged.submodules["query_text_encoder"]
    assert len(text.interior_nodes) == 2
    order = {n: i for i, n in enumerate(tg.graph_module.graph.nodes)}
    assert [order[n] for n in text.interior_nodes] == sorted(order[n] for n in text.interior_nodes)

    # coverage: every original leaf is accounted for exactly once.
    covered = [u for s in plan.partitions.values() for u in s]
    assert sorted(covered) == sorted(tg.submodules)

    # the merged graph still reads as the diamond (merged uid maps to the text role).
    assert partition.validate_diamond(merged).is_diamond


def test_apply_recomputes_dep_graph():
    tg = _diamond_tg()
    plan = SplitPlan(partitions={uid: frozenset({uid}) for uid in tg.submodules}, objective_value=0.0)
    merged = partition.apply(tg, plan)
    # identity plan (one partition per leaf) reproduces the original dep_graph.
    assert {k: set(v) for k, v in merged.dep_graph.items()} == {k: set(v) for k, v in tg.dep_graph.items()}
