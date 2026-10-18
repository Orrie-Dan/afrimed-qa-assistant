"""Batched greedy generation, shared by evaluation and the app."""
import torch

from .prompts import to_messages

# Qwen2.5's own generation_config uses repetition_penalty 1.05; it is kept, but made explicit.
GEN_KWARGS = {"do_sample": False, "repetition_penalty": 1.05, "use_cache": True}


def build_prompt(tok, rec: dict) -> str:
    return tok.apply_chat_template(to_messages(rec, with_answer=False), tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int = 16) -> list[str]:
    """Greedy-decode each prompt; returns texts in input order."""
    model.eval()
    old_side, tok.padding_side = tok.padding_side, "left"
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # similar lengths per batch
    out = [None] * len(prompts)
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                                 **GEN_KWARGS)
            for i, row in zip(idx, gen[:, enc["input_ids"].shape[1]:]):
                out[i] = tok.decode(row, skip_special_tokens=True).strip()
    finally:
        tok.padding_side = old_side
    return out
