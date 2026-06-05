import pytest

import vortexsplit.patches as patches


@pytest.fixture(scope="session")
def device():
    from torch import device
    from torch.cuda import is_available

    return device("cuda" if is_available() else "cpu")


@pytest.fixture(scope="session")
def qa_dataset():
    from datasets import load_dataset

    return load_dataset(
        "BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
        "OKVQA_data",
        split="test",
    )


@pytest.fixture(scope="session")
def image_dataset():
    from datasets import load_dataset

    return load_dataset("detection-datasets/coco", split="val")


@pytest.fixture(scope="session")
def query_sample(qa_dataset, image_dataset):
    """A single (texts, images) query drawn from the benchmark datasets.

    Both are lists because the wrapper batches: ``texts[i]`` pairs with
    ``images[i]``.
    """
    sample = qa_dataset[0]
    coco_sample = next(s for s in image_dataset if s["image_id"] == sample["img_key"])
    image = coco_sample["image"].convert("RGB")
    text = sample["instruction"] + sample["question"]
    return [text], [image]


@pytest.fixture(scope="session")
def image_processor():
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")


@pytest.fixture(scope="session")
def preflmr_forward_and_model(device):
    model = patches.flmr.get_model(device=device)

    def forward(input_ids, attention_mask, pixel_values):
        return model.query(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    return forward, model


@pytest.fixture(scope="session")
def preflmr_patched_forward_and_model(device):
    model = patches.flmr.get_model(device=device)
    model = patches.flmr.patch_model(model)

    def forward(input_ids, attention_mask, pixel_values):
        return model.query(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

    return forward, model


@pytest.fixture(scope="session")
def query_inputs(query_sample, image_processor, preflmr_patched_forward_and_model, device):
    """The (input_ids, attention_mask, pixel_values) tensors for one real query."""
    _, model = preflmr_patched_forward_and_model
    texts, images = query_sample
    encoded = model.query_tokenizer(texts)
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        image_processor(images, return_tensors="pt").pixel_values.to(device),
    )


@pytest.fixture(scope="session")
def multi_query_inputs(qa_dataset, image_dataset, image_processor, preflmr_patched_forward_and_model, device):
    """Several real queries as batch-1 (input_ids, attention_mask, pixel_values).

    All texts are tokenized together so they pad to a common length, then sliced to
    batch 1 — so every sample shares the shapes a batch-1 trace bakes in, while the
    token values, images, and (value-dependent) query masks differ across samples.
    """
    _, model = preflmr_patched_forward_and_model
    coco_index = {img_id: i for i, img_id in enumerate(image_dataset["image_id"])}
    texts: list[str] = []
    images = []
    for sample in qa_dataset:
        idx = coco_index.get(sample["img_key"])
        if idx is None:
            continue
        texts.append(sample["instruction"] + sample["question"])
        images.append(image_dataset[idx]["image"].convert("RGB"))
        if len(texts) == 5:
            break

    encoded = model.query_tokenizer(texts)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = image_processor(images, return_tensors="pt").pixel_values.to(device)
    return [(input_ids[i : i + 1], attention_mask[i : i + 1], pixel_values[i : i + 1]) for i in range(len(texts))]


@pytest.fixture(scope="session")
def traced_graph(preflmr_patched_forward_and_model, query_inputs, device):
    """A VortexGraph of the patched PreFLMR query(), traced at this query's shapes."""
    from vortexsplit.core import trace

    _, model = preflmr_patched_forward_and_model
    input_ids, attention_mask, pixel_values = query_inputs

    def forward(input_ids, attention_mask, pixel_values):
        return model.query(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        ).late_interaction_output

    return trace(forward, input_ids, attention_mask, pixel_values, original_module=model, tracing_mode="real")
