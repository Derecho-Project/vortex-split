"""Make FLMR's query mask trace symbolically instead of baking a constant.

These are drop-in replacements that compute the identical mask but are not folded
during trace-time.
"""

import types

import torch
from transformers import AutoModel, AutoTokenizer


def _mask(self, input_ids, skiplist):
    """Symbolic form of FLMR.mask: keep = (token != pad) and (token not in skiplist)."""
    keep = input_ids != 0
    skip_ids = [t for t in skiplist if isinstance(t, int)]
    if skip_ids:
        skip = torch.as_tensor(skip_ids, device=input_ids.device)
        keep = keep & ~torch.isin(input_ids, skip)
    return keep  # (B, L) bool — was list[list[bool]] upstream


def _query_mask(self, input_ids, skiplist):
    """Symbolic form of FLMR.query_mask: also drop tokens at/inside the instruction."""
    keep = _mask(self, input_ids, skiplist)
    if not self.mask_instruction:
        return keep
    sep_id = self.instruction_token_id
    sep = torch.argmax((input_ids == sep_id).int(), dim=1).clamp(min=1)  # (B,)
    # position index [0..L-1] per row, derived from input_ids so the sequence
    # extent tracks the input dimension instead of baking a trace-time arange(L).
    pos = torch.cumsum(torch.ones_like(input_ids), dim=1) - 1  # (B, L)
    after_instruction = (pos > sep[:, None]) | (pos < 2)
    return keep & after_instruction


def get_model(
    checkpoint: str = "LinWeizheDragon/PreFLMR_ViT-L",
    device: torch.device = torch.device("cpu"),
):
    """get the FLMR model"""

    query_tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="query_tokenizer", trust_remote_code=True)
    context_tokenizer = AutoTokenizer.from_pretrained(checkpoint, subfolder="context_tokenizer", trust_remote_code=True)

    model = AutoModel.from_pretrained(
        checkpoint,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
        trust_remote_code=True,
    ).to(device)

    return model


def patch_model(model):
    """Bind the symbolic query_mask/mask onto a live FLMR model and return it."""
    model.mask = types.MethodType(_mask, model)
    model.query_mask = types.MethodType(_query_mask, model)
    return model
