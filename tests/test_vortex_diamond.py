"""End-to-end: automatically split PreFLMR's query() and reproduce the diamond.

These exercise the whole pipeline on the real model: trace -> (plan) -> apply ->
export -> equivalence, plus the diamond topology assertion. They are slow (model
load + a full forward) and need the PreFLMR / CLIP checkpoints; run on GPU for speed.
"""

from collections import defaultdict

import pytest
import torch

from vortexsplit.core import export, optimize, partition, trace
from vortexsplit.core.equivalence import assert_equivalent, check_equivalence, check_equivalence_over
from vortexsplit.core.models import SplitPlan
from vortexsplit.core.profile import static_profile


def _inputs_by_name(tg, query_inputs) -> dict[str, torch.Tensor]:
    """Map the traced graph's placeholders (in order) to the query tensors."""
    names = [n.name for n in tg.graph_module.graph.nodes if n.op == "placeholder"]
    assert len(names) == len(query_inputs), (names, len(query_inputs))
    return dict(zip(names, query_inputs))


def _depth2_plan(tg) -> SplitPlan:
    """Group leaves by their depth-2 FQN prefix — the natural diamond units."""

    def top(uid: str) -> str:
        parts = uid.split(".")
        return ".".join(parts[:2]) if len(parts) > 2 else uid

    groups: dict[str, set[str]] = defaultdict(set)
    for uid in tg.submodules:
        groups[top(uid)].add(uid)
    return SplitPlan(partitions={k: frozenset(v) for k, v in groups.items()}, objective_value=0.0)


@pytest.mark.slow
def test_leaf_graph_is_the_diamond(traced_graph):
    """The raw traced graph already exhibits the A/B/C/D dependency topology."""
    report = partition.validate_diamond(traced_graph)
    assert report.is_diamond, report.detail


@pytest.mark.slow
def test_depth2_split_reproduces_diamond_and_output(traced_graph, query_inputs):
    tg = traced_graph
    inputs = _inputs_by_name(tg, query_inputs)
    plan = _depth2_plan(tg)

    merged = partition.apply(tg, plan)
    assert partition.validate_diamond(merged).is_diamond, partition.validate_diamond(merged).detail

    artifact = export.export(tg, plan)
    result = assert_equivalent(tg, artifact, inputs)  # bit-exact or raises with the culprit
    assert result.equal
    assert result.max_abs_diff == 0.0


@pytest.mark.slow
def test_dp_split_is_equivalent(traced_graph, query_inputs):
    """The optimizer's own plan (profiled costs) also round-trips bit-exactly."""
    tg = traced_graph
    inputs = _inputs_by_name(tg, query_inputs)

    static_profile(tg)
    # one cheap timing pass so the bottleneck DP has compute costs to chew on.
    from vortexsplit.core.profile import dynamic_profile

    batch = query_inputs[0].shape[0]
    dynamic_profile(tg, lambda _b: query_inputs, batch_sizes=(batch,), trials=1, warmup=0)

    plan = optimize.solve(tg, batch_size=batch)
    artifact = export.export(tg, plan)
    assert check_equivalence(tg, artifact, inputs).equal


@pytest.mark.slow
def test_equivalence_across_multiple_inputs(preflmr_patched_forward_and_model, multi_query_inputs):
    """The split must reproduce the monolith on many different queries, not just one.

    Different samples drive different (value-dependent) query masks through Step D, so
    this fuzzes the split's correctness across real inputs sharing the traced shapes.
    """
    _, model = preflmr_patched_forward_and_model
    samples = multi_query_inputs
    assert len(samples) >= 2

    def forward(input_ids, attention_mask, pixel_values):
        return model.query(
            input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values
        ).late_interaction_output

    tg = trace(forward, *samples[0], original_module=model, tracing_mode="real")
    artifact = export.export(tg, _depth2_plan(tg))

    names = [n.name for n in tg.graph_module.graph.nodes if n.op == "placeholder"]
    inputs_list = [dict(zip(names, sample)) for sample in samples]

    result = check_equivalence_over(tg, artifact, inputs_list)
    assert result.equal, result.detail
    assert result.max_abs_diff == 0.0
    assert len(result.per_output_diff) == len(samples)


@pytest.mark.slow
def test_save_load_then_run(traced_graph, query_inputs, tmp_path):
    tg = traced_graph
    inputs = _inputs_by_name(tg, query_inputs)
    artifact = export.export(tg, _depth2_plan(tg))
    artifact.save(tmp_path / "flmr.tspart")
    reloaded = export.SplitArtifact.load(tmp_path / "flmr.tspart")
    assert check_equivalence(tg, reloaded, inputs).equal
