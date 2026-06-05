"""VortexSplit CLI — auto-segment PreFLMR's query() into exportable components."""

import argparse
import os
from pathlib import Path
from typing import cast

import torch
from datasets import Dataset, load_dataset
from flmr import (  # type: ignore
    create_searcher,
    search_custom_collection,
)
from transformers import CLIPImageProcessor

from vortexsplit.core import export, optimize, partition, trace
from vortexsplit.core.equivalence import assert_equivalent
from vortexsplit.core.export import SplitArtifact
from vortexsplit.core.profile import profile
from vortexsplit.patches import flmr as patched_flmr

DEFAULT_ARTIFACT = "flmr_split.tspart"

CHECKPOINT = "LinWeizheDragon/PreFLMR_ViT-L"
IMAGE_PROCESSOR = "openai/clip-vit-large-patch14"
IMAGE_ROOT = "/data/EVQA/images"  # where fetch_datasets.py extracts the EVQA images

INDEX_ROOT = "/data/EVQA/index"
EXPERIMENT = "EVQA_train_split"
INDEX_NAME = "EVQA_PreFLMR_ViT-L"
NBITS = 8

ds = load_dataset(
    "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
    "EVQA_data",
    split="test",
)

passages = load_dataset(
    "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
    "EVQA_passages",
    split="test_passages",
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(device: torch.device):
    class Monolithic(torch.nn.Module):
        def __init__(self, flmr, image_processor):
            super().__init__()
            self.flmr = flmr
            self.image_processor = image_processor

        def forward(self, text, image):
            encoded_text = self.flmr.query_tokenizer(text)
            encoded_image = self.image_processor(image, return_tensors="pt")

            input_ids = encoded_text["input_ids"]
            attention_mask = encoded_text["attention_mask"]
            pixel_values = encoded_image.pixel_values

            return self.flmr.query(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

    m = Monolithic(
        flmr=patched_flmr.patch_model(patched_flmr.get_model(checkpoint=CHECKPOINT, device=device)),
        image_processor=CLIPImageProcessor.from_pretrained(IMAGE_PROCESSOR),
    )
    m.to(device)
    m.eval()
    return m


def _query_sample(sample_count: int = 16, seed: int = 0) -> tuple[list[str], list]:
    """Draw ``sample_count`` (text, image) query inputs from EVQA test, shuffled by ``seed``."""
    from PIL import Image

    rows = cast(Dataset, ds).shuffle(seed=seed).select(range(sample_count))
    texts: list[str] = []
    images: list = []
    for row in cast("list[dict]", rows):
        texts.append(f"{row['instruction'].strip()} {row['question'].strip()}")
        path = os.path.join(IMAGE_ROOT, row["img_path"]) if row["img_path"] else None
        if path and os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            raise RuntimeError("image not found")
    return texts, images


def _preprocess(model, text, images, device):
    """Tokenize + image-process OUTSIDE the traced region (these are not torch ops).

    Returns the (input_ids, attention_mask, pixel_values) tensors that ARE the real
    inputs to query() — so the split graph has clean placeholders instead of baked
    constants. The query tokenizer pads to a fixed length, so only batch/seq shape is
    specialized, never the token values.
    """
    encoded = model.flmr.query_tokenizer(text)
    pixel_values = model.image_processor(images, return_tensors="pt").pixel_values
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        pixel_values.to(device),
    )


def _forward(flmr_model):
    """The tensor->tensor computation to split: query() only, no preprocessing."""

    def forward(input_ids, attention_mask, pixel_values):
        return flmr_model.query(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        ).late_interaction_output

    return forward


def plot_results(text, images, ranking_dict, passage_ids, out_path="retrieval.png", passage_contents=None, title=None):
    """Tile the queries in a grid: each cell is the query image + question above a
    table of its top-k retrieved passages (rank, score, passage)."""
    import math

    import matplotlib

    matplotlib.use("Agg")  # headless: write a file instead of opening a window
    import matplotlib.pyplot as plt

    n = len(text)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 4.8, rows * 4.4))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    subfigs = fig.subfigures(rows, cols, squeeze=False)

    for i in range(n):
        sf = subfigs[i // cols][i % cols]
        ax_img, ax_tab = sf.subplots(2, 1, gridspec_kw={"height_ratios": [3, 2]})
        ax_img.imshow(images[i])
        ax_img.axis("off")
        ax_img.set_title(f"Q{i}: {text[i][:60]}", fontsize=8)

        ax_tab.axis("off")
        cell_text = []
        for rank, (passage_idx, _r, score) in enumerate(sorted(ranking_dict[i], key=lambda r: -r[2]), start=1):
            passage = (
                passage_contents[passage_idx][:46].replace("\n", " ")
                if passage_contents is not None
                else passage_ids[passage_idx][:36]
            )
            cell_text.append([str(rank), f"{score:.2f}", passage])
        table = ax_tab.table(
            cellText=cell_text,
            colLabels=["#", "score", "passage"],
            colWidths=[0.08, 0.16, 0.76],
            cellLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.35)
        for col in range(3):  # tint + bold the header row
            table[0, col].set_facecolor("#dde6f5")
            table[0, col].set_text_props(fontweight="bold")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def _make_searcher():
    return create_searcher(
        index_root_path=INDEX_ROOT,
        index_experiment_name=EXPERIMENT,
        index_name=INDEX_NAME,
        nbits=NBITS,
        use_gpu=torch.cuda.is_available(),
    )


def _retrieve(searcher, text, query_embeddings, k: int = 5):
    """Run ColBERT late-interaction search for ``query_embeddings`` -> ranking dict."""
    queries = {i: t for i, t in enumerate(text)}
    ranking = search_custom_collection(
        searcher=searcher,
        queries=queries,
        query_embeddings=query_embeddings,
        num_document_to_retrieve=k,
        remove_zero_tensors=True,  # required for PreFLMR
    )
    return ranking.todict()


def baseline(text, images, k: int = 5):
    model = _load_model(_device())
    with torch.no_grad():
        out = model.forward(text, images)
    # PreFLMR's query() returns late_interaction_output: (B, query_len, dim).
    # ColBERT search runs on CPU, so detach + move there.
    query_embeddings = out.late_interaction_output.detach().cpu()

    searcher = create_searcher(
        index_root_path=INDEX_ROOT,
        index_experiment_name=EXPERIMENT,
        index_name=INDEX_NAME,
        nbits=NBITS,
        use_gpu=torch.cuda.is_available(),
    )

    queries = {i: t for i, t in enumerate(text)}
    ranking = search_custom_collection(
        searcher=searcher,
        queries=queries,
        query_embeddings=query_embeddings,
        num_document_to_retrieve=k,
        remove_zero_tensors=True,  # required for PreFLMR
    )
    ranking_dict = ranking.todict()

    passage_contents = list(passages["passage_content"])
    passage_ids = list(passages["passage_id"])
    for i in range(len(text)):
        print(f"\nQ{i}: {text[i]}")
        for passage_idx, _rank, score in ranking_dict[i]:
            snippet = passage_contents[passage_idx][:80].replace("\n", " ")
            print(f"   {score:7.2f}  {passage_ids[passage_idx]:34s} {snippet}...")

    plot_results(text, images, ranking_dict, passage_ids)
    return ranking_dict


def cmd_generate(args: argparse.Namespace) -> None:
    print("here")
    device = _device()
    print(f"loading PreFLMR on {device} ...")
    model = _load_model(device)

    print("tracing query() and profiling ...")
    text, images = _query_sample(sample_count=args.batch)
    inputs = _preprocess(model, text, images, device)  # tensors, computed OUTSIDE the graph
    bsize = inputs[0].shape[0]

    # Split query() only (tensor -> tensor), rooted at the FLMR model so submodule
    # FQNs are query_text_encoder.* etc. (not flmr.query_text_encoder.*).
    tg = trace(_forward(model.flmr), *inputs, original_module=model.flmr, tracing_mode="real")
    profile(tg, lambda _b: inputs, batch_sizes=(bsize,), trials=args.trials, warmup=1, device=device)

    print(f"optimizing split (max_partitions={args.max_partitions}) ...")
    plan = optimize.solve(tg, batch_size=bsize, max_partitions=args.max_partitions)
    if args.coarsen:
        # second pass: coarsen the per-module split into the Vortex A/B/C/D stages
        # by modality provenance (text / vision / fusion) + FQN family.
        plan = optimize.collapse_by_modality(tg, plan)
    merged = partition.apply(tg, plan)
    diamond = partition.validate_diamond(merged)

    artifact = export.export(tg, plan)
    placeholder_names = [n.name for n in tg.graph_module.graph.nodes if n.op == "placeholder"]
    equiv = assert_equivalent(tg, artifact, dict(zip(placeholder_names, inputs)))

    out = Path(args.out)
    artifact.save(out)
    try:
        merged.draw(str(out / "diagram.svg"), depth=5)
    except Exception as exc:  # graphviz `dot` not installed — non-fatal
        print(f"(skipped diagram: {exc})")

    print(
        f"\nsplit into {len(plan.partitions)} partitions at batch {bsize} | "
        f"diamond: {diamond.detail} | equivalence: {equiv.detail}"
    )
    for uid in sorted(plan.partitions):
        print(f"  {uid[:58]:58s} <- {len(plan.partitions[uid]):3d} leaves")


def cmd_demo(args: argparse.Namespace) -> None:
    device = _device()
    text, images = _query_sample(sample_count=args.batch, seed=args.sample_seed)

    print(f"loading PreFLMR on {device} for the monolith pass ...")
    model = _load_model(device)
    inputs = _preprocess(model, text, images, device)
    with torch.no_grad():
        mono_emb = _forward(model.flmr)(*inputs).detach().cpu()

    print(f"loading the split from {args.artifact}/ ...")
    artifact = SplitArtifact.load(args.artifact).to(device).eval()
    named_inputs = dict(zip(artifact.manifest.graph_input_names, inputs))
    try:
        (split_emb,) = artifact.run(**named_inputs)
    except Exception as exc:
        raise SystemExit(
            f"running the split failed ({exc}); the artifact was traced at a fixed batch — "
            f"pass the same --batch you used to generate it (this run used batch {inputs[0].shape[0]})."
        ) from exc
    split_emb = split_emb.detach().cpu()
    print(f"query_embeddings max|diff| (split vs monolith): {(split_emb - mono_emb).abs().max().item():.3e}")

    passage_ids = list(passages["passage_id"])
    passage_contents = list(passages["passage_content"])
    searcher = _make_searcher()
    plot_results(
        text,
        images,
        _retrieve(searcher, text, mono_emb, k=args.k),
        passage_ids,
        out_path=args.baseline_out,
        passage_contents=passage_contents,
        title="monolith (unexported)",
    )
    plot_results(
        text,
        images,
        _retrieve(searcher, text, split_emb, k=args.k),
        passage_ids,
        out_path=args.split_out,
        passage_contents=passage_contents,
        title="split (from artifact)",
    )
    print(f"wrote {args.baseline_out} (monolith) and {args.split_out} (split)")


def cmd_draw(args: argparse.Namespace) -> None:
    from vortexsplit.core.export import load_manifest, render_flow

    manifest = load_manifest(args.artifact)  # reads manifest.json only — no weights
    dst = render_flow(manifest, args.out)
    print(f"wrote {dst} ({len(manifest.specs)} stages: {' -> '.join(manifest.topo_order)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="trace, segment, and export the partitioned model")
    gen.add_argument("--out", default=DEFAULT_ARTIFACT, help="output directory for the split")
    gen.add_argument("--max-partitions", type=int, default=8, help="cap on the number of stages from the first pass")
    gen.add_argument("--batch", type=int, default=16, help="batch size to trace/profile the split at")
    gen.add_argument("--trials", type=int, default=3, help="profiling trials for the cost model")
    gen.add_argument("--coarsen", action="store_true", help="second pass to further coarsen the split")
    gen.set_defaults(func=cmd_generate)

    demo = sub.add_parser("demo", help="retrieval through the monolith vs the loaded split -> two PNGs")
    demo.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="the split directory to run")
    demo.add_argument("--batch", type=int, default=16, help="number of queries (must match the artifact's trace batch)")
    demo.add_argument("--sample-seed", type=int, default=0, help="seed the sample")
    demo.add_argument("--k", type=int, default=5, help="passages to retrieve per query")
    demo.add_argument("--baseline-out", default="retrieval_baseline.png", help="monolith retrieval plot")
    demo.add_argument("--split-out", default="retrieval_split.png", help="split retrieval plot")
    demo.set_defaults(func=cmd_demo)

    drw = sub.add_parser("draw", help="render the manifest's partition dataflow to SVG")
    drw.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="the split directory to read manifest.json from")
    drw.add_argument("--out", default="flow.svg", help="output image path (.svg/.pdf)")
    drw.set_defaults(func=cmd_draw)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
