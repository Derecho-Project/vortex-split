"""Profile a traced VortexGraph: fill the per-component cost fields.

static_profile  — sums each component's parameter count and resident weight bytes
                  from its weight (get_attr) operators. No execution.
dynamic_profile — per (batch_size, trial): one timed FX Interpreter pass over the
                  whole DAG for per-component compute time, capturing the tensors
                  that cross each component boundary; then each component is replayed
                  alone on those tensors while PYNVML samples the device-memory
                  high-water mark — the GPU memory budget the component needs.

profile() runs both and returns the same (mutated) graph.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.fx as fx

from vortexsplit.core.models import VortexGraph


def _resolve_get_attr(gm: fx.GraphModule, target: str) -> torch.Tensor | None:
    """Resolve a (possibly dotted) get_attr target to its tensor, or None."""
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.Tensor) else None


def _extract_partition(gm: fx.GraphModule, sub: VortexGraph.Submodule) -> fx.GraphModule:
    """Standalone GraphModule for one component, runnable on its real boundary tensors.

    Input operators become placeholders; interior operators are copied verbatim (so
    interior get_attr operators keep their target and resolve against the parent gm,
    passed as root); output operators become the return.
    """
    g = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for inp in sub.input_nodes:
        env[inp] = g.placeholder(inp.name.replace(".", "_"))
    for n in sub.interior_nodes:
        env[n] = g.node_copy(n, lambda x: env[x])
    g.output(tuple(env[o] for o in sub.output_nodes))
    return fx.GraphModule(gm, g)


# --- PYNVML device-memory sampling -----------------------------------------
_NVML_INITED = False


def _nvml_handle(device: torch.device):
    """PYNVML handle for ``device``, or None if pynvml/NVML is unavailable. NVML
    enumerates by PCI order; we assume that matches torch's device index."""
    global _NVML_INITED
    try:
        import pynvml
    except ImportError:
        return None
    if not _NVML_INITED:
        pynvml.nvmlInit()
        _NVML_INITED = True
    index = device.index if device.index is not None else torch.cuda.current_device()
    return pynvml.nvmlDeviceGetHandleByIndex(index)


class _MemorySampler:
    """Daemon thread that polls device-memory usage as fast as NVML allows while a
    component runs; ``peak_bytes`` is the high-water mark above the pre-run baseline."""

    def __init__(self, handle):
        import pynvml

        self._used = lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used
        self._stop = threading.Event()
        self._baseline = self._used()
        self._peak = self._baseline
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self._stop.is_set():
            used = self._used()
            if used > self._peak:
                self._peak = used

    def __enter__(self) -> _MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_bytes(self) -> int:
        return max(0, self._peak - self._baseline)  # type: ignore


def _peak_activation_bytes(subgm: fx.GraphModule, inputs: tuple, device: torch.device) -> int:
    """Device-memory high-water mark PYNVML observes while ``subgm`` runs alone on its
    captured inputs — the component's real GPU footprint. 0 off CUDA or without NVML."""
    handle = _nvml_handle(device) if device.type == "cuda" else None
    if handle is None:
        with torch.no_grad():
            subgm(*inputs)
        return 0
    torch.cuda.synchronize()
    with _MemorySampler(handle) as sampler:
        with torch.no_grad():
            subgm(*inputs)
        torch.cuda.synchronize()
    return sampler.peak_bytes


def static_profile(tg: VortexGraph) -> VortexGraph:
    """Fill model_params / model_static_size_byte from each component's weight operators."""
    gm = tg.graph_module
    for sub in tg.submodules.values():
        params = 0
        nbytes = 0
        seen: set[int] = set()
        for node in sub.interior_nodes:
            if node.op != "get_attr":
                continue
            t = _resolve_get_attr(gm, str(node.target))
            if t is None:
                continue
            key = t.untyped_storage().data_ptr() if t.is_contiguous() else id(t)
            if key in seen:
                continue
            seen.add(key)
            params += t.numel()
            nbytes += t.numel() * t.element_size()
        sub.model_params = params
        sub.model_static_size_byte = nbytes
    return tg


class _NodeProfiler(fx.Interpreter):
    """Times each operator and captures the outputs of a chosen set of them."""

    def __init__(
        self,
        gm: fx.GraphModule,
        device: torch.device,
        capture: set[fx.Node] | None = None,
    ):
        super().__init__(gm)
        self._cuda = device.type == "cuda"
        self._capture = capture or set()
        self.node_ns: dict[fx.Node, int] = {}
        self.captured: dict[fx.Node, object] = {}
        self._events: list[tuple[fx.Node, torch.cuda.Event, torch.cuda.Event]] = []

    def run_node(self, n: fx.Node):
        if self._cuda:
            # cast: torch types Event.__new__ as the parent whose record() wants a stream.
            start = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
            end = cast(torch.cuda.Event, torch.cuda.Event(enable_timing=True))
            start.record()
            result = super().run_node(n)
            end.record()
            self._events.append((n, start, end))
        else:
            import time

            t0 = time.perf_counter_ns()
            result = super().run_node(n)
            self.node_ns[n] = time.perf_counter_ns() - t0
        if n in self._capture:
            self.captured[n] = result
        return result

    def run(self, *args, **kwargs):
        out = super().run(*args, **kwargs)
        if self._cuda:
            torch.cuda.synchronize()  # one sync, not per-op, so timing isn't serialized
            for n, start, end in self._events:
                self.node_ns[n] = int(start.elapsed_time(end) * 1e6)  # ms float -> ns
            self._events.clear()
        return out


def dynamic_profile(
    tg: VortexGraph,
    build_inputs: Callable[[int], Sequence],
    batch_sizes: Sequence[int] = (1,),
    *,
    trials: int = 10,
    warmup: int = 3,
    device: torch.device | None = None,
) -> VortexGraph:
    """Fill model_runtime_ns (whole-DAG operator timing) and model_runtime_size_bytes
    (PYNVML device-memory peak per component)."""
    if device is None:
        device = next(tg.graph_module.parameters()).device

    gm = tg.graph_module
    gm.eval()

    owner: dict[fx.Node, str] = {n: uid for uid, sub in tg.submodules.items() for n in sub.interior_nodes}
    boundary: set[fx.Node] = {n for sub in tg.submodules.values() for n in sub.input_nodes}
    subgms = {uid: _extract_partition(gm, sub) for uid, sub in tg.submodules.items()}

    for sub in tg.submodules.values():
        sub.model_runtime_ns = {b: [] for b in batch_sizes}
        sub.model_runtime_size_bytes = {b: [] for b in batch_sizes}

    with torch.no_grad():
        for b in batch_sizes:
            inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in build_inputs(b))

            for _ in range(warmup):
                _NodeProfiler(gm, device).run(*inputs)

            for _ in range(trials):
                prof = _NodeProfiler(gm, device, capture=boundary)
                prof.run(*inputs)

                ns_by_uid: dict[str, int] = {u: 0 for u in tg.submodules}
                for n, ns in prof.node_ns.items():
                    uid = owner.get(n)
                    if uid is not None:
                        ns_by_uid[uid] += ns

                for uid, sub in tg.submodules.items():
                    sub.model_runtime_ns[b].append(ns_by_uid[uid])
                    part_inputs = tuple(prof.captured[n] for n in sub.input_nodes)
                    peak = _peak_activation_bytes(subgms[uid], part_inputs, device)
                    sub.model_runtime_size_bytes[b].append(peak)

                prof.captured.clear()  # release boundary tensors before the next trial
    return tg


def profile(
    tg: VortexGraph,
    build_inputs: Callable[[int], Sequence],
    batch_sizes: Sequence[int] = (1,),
    **kwargs,
) -> VortexGraph:
    """Run both passes; returns the same (mutated) VortexGraph."""
    static_profile(tg)
    dynamic_profile(tg, build_inputs, batch_sizes, **kwargs)
    return tg
