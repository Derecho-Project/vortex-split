import pytest
import torch

N_SAMPLES = 500


@pytest.fixture(scope="session")
def coco_index(image_dataset):
    """Map COCO ``image_id -> row index`` once, without decoding any image."""
    return {img_id: i for i, img_id in enumerate(image_dataset["image_id"])}


@pytest.fixture(scope="session")
def multi_query_samples(qa_dataset, image_dataset, coco_index):
    """The first ``N_SAMPLES`` benchmark queries whose image is in the COCO
    split, each as a single-item ``(texts, images)`` pair."""
    samples = []
    for sample in qa_dataset:
        idx = coco_index.get(sample["img_key"])
        if idx is None:
            continue
        image = image_dataset[idx]["image"].convert("RGB")
        samples.append(([sample["instruction"] + sample["question"]], [image]))
        if len(samples) == N_SAMPLES:
            break
    assert len(samples) == N_SAMPLES, f"only found {len(samples)}/{N_SAMPLES} samples"
    return samples


@pytest.mark.parametrize("sample_idx", range(N_SAMPLES))
def test_patch_zero_regression(
    sample_idx,
    multi_query_samples,
    preflmr_forward_and_model,
    preflmr_patched_forward_and_model,
    image_processor,
    device,
):
    """regression test to ensure patched FLMR behaves the same"""
    default_forward, model = preflmr_forward_and_model
    patched_forward, _ = preflmr_patched_forward_and_model

    # Build the pure-tensor inputs from the real sample (the preprocessing
    # query() expects: query-tokenized text + CLIP-processed image).
    texts, images = multi_query_samples[sample_idx]
    encoded = model.query_tokenizer(texts)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = image_processor(images, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        default = default_forward(input_ids, attention_mask, pixel_values)
        patched = patched_forward(input_ids, attention_mask, pixel_values)

    default = default.late_interaction_output
    patched = patched.late_interaction_output

    assert patched.shape == default.shape, (default.shape, patched.shape)
    assert torch.equal(patched, default), (
        f"sample {sample_idx} (seq_len={input_ids.shape[1]}): patched query() "
        f"diverged from default, max|diff|={(patched - default).abs().max().item():.3e}"
    )
