"""
python fetch_datasets.py                         # both modalities -> /data/EVQA/images
python fetch_datasets.py --modality landmark     # cheaper: 2.79 GB, 1750 queries
python fetch_datasets.py --image-root /data/EVQA/images --keep-archives
"""

import argparse
import os
import tarfile
import zipfile

DATA_REPO = "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR"
IMAGE_REPO = "BByrneLab/M2KR_Images"

# modality -> (archive in M2KR_Images, top-level dir it extracts to, img_path prefix)
ARCHIVES = {
    "inat": ("EVQA/inat.zip", "inat"),
    "landmark": ("EVQA/google-landmark.tar", "google-landmark"),
}


def fetch_text():
    """Cache EVQA_data (all splits) and EVQA_passages so retrieval can run offline."""
    from datasets import load_dataset

    print("== M2KR text: EVQA_data + EVQA_passages ==")
    for split in ("train", "valid", "test"):
        ds = load_dataset(DATA_REPO, "EVQA_data", split=split)
        print(f"   EVQA_data[{split}]: {len(ds)} queries")
    for split in ("train", "valid", "test"):
        ds = load_dataset(DATA_REPO, "EVQA_passages", split=f"{split}_passages")
        print(f"   EVQA_passages[{split}]: {len(ds)} passages")


def already_extracted(image_root, top_dir):
    d = os.path.join(image_root, top_dir)
    return os.path.isdir(d) and any(os.scandir(d))


def _has_prefix(names, prefix):
    """True if the archive members already live under ``prefix/`` (no wrapper needed)."""
    return any(n.lstrip("./").startswith(prefix + "/") for n in names[:20])


def extract(archive_path, image_root, prefix):
    """Extract so that members land under ``image_root/prefix/`` to match img_path.

    The archives ship their contents flat (inat.zip -> val/..., google-landmark.tar ->
    train/...), but img_path is ``inat/val/...`` / ``google-landmark/train/...`` — so we
    extract into the ``prefix`` subdir. If an archive already includes that prefix, we
    extract at the root instead to avoid double-nesting.
    """
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as z:  # tolerant of leading/trailing bytes
            names = z.namelist()
            target = image_root if _has_prefix(names, prefix) else os.path.join(image_root, prefix)
            os.makedirs(target, exist_ok=True)
            print(f"   extracting {os.path.basename(archive_path)} -> {target}/ ...")
            z.extractall(target)
    else:
        with tarfile.open(archive_path) as t:  # trusted source
            names = t.getnames()
            target = image_root if _has_prefix(names, prefix) else os.path.join(image_root, prefix)
            os.makedirs(target, exist_ok=True)
            print(f"   extracting {os.path.basename(archive_path)} -> {target}/ ...")
            t.extractall(target)


def fetch_images(modalities, image_root, keep_archives):
    from huggingface_hub import hf_hub_download

    raw_dir = os.path.join(image_root, "_archives")
    os.makedirs(image_root, exist_ok=True)
    for m in modalities:
        archive, top_dir = ARCHIVES[m]
        print(f"== EVQA images: {m} ({archive}) ==")
        if already_extracted(image_root, top_dir):
            print(f"   {top_dir}/ already present — skipping")
            continue
        print("   downloading (resumable) ...")
        local = hf_hub_download(IMAGE_REPO, archive, repo_type="dataset", local_dir=raw_dir)
        extract(local, image_root, top_dir)
        if not keep_archives:
            os.remove(local)
            print(f"   removed archive {os.path.basename(local)}")


def verify(image_root):
    """Report how many EVQA test img_paths now resolve on disk."""
    from datasets import load_dataset

    ds = load_dataset(DATA_REPO, "EVQA_data", split="test")
    have = miss = 0
    for p in ds["img_path"]:
        if os.path.exists(os.path.join(image_root, p)):
            have += 1
        else:
            miss += 1
    print(f"\n== verify == {have}/{have + miss} EVQA test images resolve under {image_root}/")
    if miss:
        print(f"   ({miss} missing — expected if you fetched only one modality)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default="/data/EVQA/images", help="extract images here (mirrors img_path)")
    ap.add_argument("--modality", choices=["inat", "landmark", "both"], default="both")
    ap.add_argument("--keep-archives", action="store_true", help="keep the downloaded .zip/.tar after extracting")
    ap.add_argument("--skip-text", action="store_true", help="don't (re)cache EVQA_data/passages")
    args = ap.parse_args()

    mods = ["inat", "landmark"] if args.modality == "both" else [args.modality]
    if not args.skip_text:
        fetch_text()
    fetch_images(mods, args.image_root, args.keep_archives)
    verify(args.image_root)
    print("\ndone. run:")
    print(f"  HF_HUB_OFFLINE=1 uv run main.py")


if __name__ == "__main__":
    main()
