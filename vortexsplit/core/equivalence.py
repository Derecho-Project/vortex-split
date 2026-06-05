"""Prove that an exported split reproduces the monolithic graph's output."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vortexsplit.core import profile
from vortexsplit.core.export import SplitArtifact
from vortexsplit.core.models import VortexGraph


@dataclass
class EquivalenceResult:
    equal: bool
    max_abs_diff: float
    per_output_diff: list[float] = field(default_factory=list)
    detail: str = ""

    def __bool__(self) -> bool:
        return self.equal


def _positional(tg: VortexGraph, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    names = [n.name for n in tg.graph_module.graph.nodes if n.op == "placeholder"]
    missing = set(names) - set(inputs)
    if missing:
        raise ValueError(f"missing graph inputs {sorted(missing)}")
    return tuple(inputs[n] for n in names)


def _as_tuple(out) -> tuple:
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def check_equivalence(
    tg: VortexGraph,
    artifact: SplitArtifact,
    inputs: dict[str, torch.Tensor],
    *,
    exact: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> EquivalenceResult:
    """Compare the split's output against the traced monolith on ``inputs``."""
    tg.graph_module.eval()
    with torch.no_grad():
        reference = _as_tuple(tg.graph_module(*_positional(tg, inputs)))
    produced = artifact.run(**inputs)

    if len(reference) != len(produced):
        return EquivalenceResult(False, float("inf"), detail=f"arity {len(reference)} vs {len(produced)}")

    diffs = [(p - r).abs().max().item() if r.numel() else 0.0 for r, p in zip(reference, produced)]
    max_diff = max(diffs, default=0.0)
    if exact:
        ok = all(torch.equal(r, p) for r, p in zip(reference, produced))
    else:
        ok = all(torch.allclose(r, p, rtol=rtol, atol=atol) for r, p in zip(reference, produced))
    detail = "bit-identical" if (exact and ok) else f"max|diff|={max_diff:.3e}"
    return EquivalenceResult(ok, max_diff, diffs, detail)


def check_equivalence_over(
    tg: VortexGraph,
    artifact: SplitArtifact,
    inputs_list: list[dict[str, torch.Tensor]],
    *,
    exact: bool = True,
) -> EquivalenceResult:
    """Check the split against the monolith across several inputs at once."""
    if not inputs_list:
        raise ValueError("check_equivalence_over needs at least one input")
    ok = True
    worst = 0.0
    per_input: list[float] = []
    failures: list[str] = []
    for i, inputs in enumerate(inputs_list):
        result = check_equivalence(tg, artifact, inputs, exact=exact)
        ok = ok and result.equal
        worst = max(worst, result.max_abs_diff)
        per_input.append(result.max_abs_diff)
        if not result.equal:
            failures.append(f"input[{i}]: {result.detail}")
    detail = "; ".join(failures) if failures else f"all {len(inputs_list)} inputs match (exact)"
    return EquivalenceResult(ok, worst, per_input, detail)


def localize(
    tg: VortexGraph,
    artifact: SplitArtifact,
    inputs: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-partition max |diff| between its isolated output and the monolith's."""
    gm = tg.graph_module
    gm.eval()
    device = next(gm.parameters()).device
    name_to_node = {n.name: n for n in gm.graph.nodes}

    capture_names: set[str] = set()
    for spec in artifact.manifest.specs:
        capture_names |= set(spec.input_node_names) | set(spec.output_node_names)
    capture = {name_to_node[nm] for nm in capture_names if nm in name_to_node}

    prof = profile._NodeProfiler(gm, device, capture=capture)
    with torch.no_grad():
        prof.run(*_positional(tg, inputs))
    captured: dict[str, object] = {node.name: val for node, val in prof.captured.items()}
    captured.update({nm: inputs[nm] for nm in artifact.manifest.graph_input_names})

    diffs: dict[str, float] = {}
    with torch.no_grad():
        for spec in artifact.manifest.specs:
            args = tuple(captured[nm] for nm in spec.input_node_names)
            outs = artifact.modules[spec.uid](*args)
            worst = 0.0
            for nm, produced in zip(spec.output_node_names, outs):
                ref = captured[nm]
                if isinstance(ref, torch.Tensor) and ref.numel():
                    worst = max(worst, (produced - ref).abs().max().item())
            diffs[spec.uid] = worst
    prof.captured.clear()
    return diffs


def assert_equivalent(
    tg: VortexGraph,
    artifact: SplitArtifact,
    inputs: dict[str, torch.Tensor],
    *,
    exact: bool = True,
) -> EquivalenceResult:
    """Raise with a localized diagnostic unless the split matches the monolith."""
    result = check_equivalence(tg, artifact, inputs, exact=exact)
    if not result.equal:
        worst = localize(tg, artifact, inputs)
        culprits = sorted(((d, uid) for uid, d in worst.items() if d > 0), reverse=True)[:5]
        blame = ", ".join(f"{uid}: {d:.3e}" for d, uid in culprits) or "no single partition diverges"
        raise AssertionError(f"split != monolith ({result.detail}); worst partitions — {blame}")
    return result
