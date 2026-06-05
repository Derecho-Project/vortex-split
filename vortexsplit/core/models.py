"""Pure data containers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch.fx as fx
import torch.nn as nn


@dataclass(frozen=True)
class SplitPlan:
    """How the optimizer groups leaf components into pipeline stages.

    ``partitions`` maps each chosen stage's uid (its region FQN) to the leaf component
    uids it absorbs; the stages form an antichain covering every leaf once.
    ``objective_value`` is the optimized cost — for the bottleneck objective, the
    slowest stage's compute+transfer time in ns.
    """

    partitions: Mapping[str, frozenset[str]]
    objective_value: float
    objective: str = "bottleneck"


@dataclass(frozen=True)
class DiamondReport:
    """Result of checking a graph against the Vortex A/B/C/D diamond.

    ``roles`` maps each component uid to its role (text/vision/step_c/step_d/output/
    other); ``role_edges`` is the role-level dependency graph; ``is_diamond`` is the
    verdict; ``detail`` explains any mismatch.
    """

    roles: Mapping[str, str]
    role_edges: frozenset[tuple[str, str]]
    is_diamond: bool
    detail: str = ""


@dataclass(frozen=True)
class PartitionSpec:
    """Serializable interface of one exported component.

    ``input_node_names`` / ``output_node_names`` are the traced graph's unique operator
    names the component takes (placeholder order) and returns (output order). Routing is
    by exact name identity, never fuzzy matching.
    """

    uid: str
    input_node_names: tuple[str, ...]
    output_node_names: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    """Wiring of an exported split: how components route tensors to form the DAG.

    ``producer_of`` maps each boundary operator name to the component that emits it;
    ``graph_input_names`` are the model-level inputs the caller supplies; ``output_names``
    are the final returned names (return order); ``topo_order`` is a valid component
    execution order (Kahn sort of the dependency graph).
    """

    specs: tuple[PartitionSpec, ...]
    producer_of: Mapping[str, str]
    graph_input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    topo_order: tuple[str, ...]


@dataclass
class ModuleRecord:
    """One nn.Module captured during a trace pass."""

    name: str
    """Attribute path in the traced model, e.g. 'encoder' or 'encoder.layer1'."""

    module: nn.Module
    """The live module object (reference into the original model)."""


@dataclass
class VortexGraph:
    """The traced DAG: operators (fx nodes) and their dataflow dependencies, grouped
    into exportable components."""

    @dataclass
    class Submodule:
        semantic_uid: str
        """Globally unique id — the innermost module FQN that issued these operators,
        e.g. 'encoder.layer.0.output.LayerNorm'."""

        module_type: str
        """Fully-qualified class name of that module, e.g.
        'torch.nn.modules.normalization.LayerNorm'."""

        interior_nodes: Sequence[fx.Node]
        """Operators issued inside this module (its membership set), topological order."""

        input_nodes: Sequence[fx.Node]
        """Producer operators outside this module that feed its interior."""

        output_nodes: Sequence[fx.Node]
        """Interior operators consumed outside this module (or by the graph output),
        topological order."""

        # --- profiled fields, filled by vortexsplit.core.profile ---
        # All default to "unprofiled" so a freshly-traced graph is valid.

        model_params: int = field(default=0)
        """Weight scalars used by this component's operators — Σ numel() over the
        distinct get_attr tensors it reads. Attributed per-operator, so it never
        double-counts across parent/child components."""

        model_static_size_byte: int = field(default=0)
        """Resident weight bytes: Σ numel()*element_size() over the same get_attr
        tensors as model_params."""

        model_runtime_size_bytes: dict[int, list[int]] = field(default_factory=dict)
        """batch_size -> per-trial activation footprint in bytes (PYNVML device-memory
        peak), one entry per trial so callers can take a mean / p95."""

        model_runtime_ns: dict[int, list[int]] = field(default_factory=dict)
        """batch_size -> per-trial device compute time in ns, summed over the
        component's operators, one entry per trial."""

    graph_module: fx.GraphModule
    """The traced fx GraphModule holding every operator."""

    submodules: Mapping[str, Submodule]
    """Exportable components keyed by semantic_uid."""

    dep_graph: Mapping[str, set[str]]
    """Component dataflow dependencies: key=source, value=set of sinks."""

    def draw(self, path: Path | str, depth: int | None = None):
        """Save a vector diagram (.pdf/.svg) to path. A component deeper than ``depth``
        is absorbed into its nearest depth-bounded ancestor; boxes show operator/param
        counts and weight/activation sizes, connected by dep_graph edges."""
        from vortexsplit.core.draw import render

        return render(self, path, depth)
