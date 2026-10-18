"""Loading the processed splits and turning records into training tensors."""
import json

import torch

from .prompts import to_messages


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def encode(tok, rec: dict, max_len: int) -> dict:
    """Token ids with labels masked (-100) on the prompt, so loss is on the answer only."""
    # Render to text first: apply_chat_template's tokenize=True return type differs across versions.
    prompt = tok.apply_chat_template(to_messages(rec, False), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(to_messages(rec), tokenize=False)
    assert full.startswith(prompt), "chat template is not prefix-stable"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids[:max_len], "labels": labels[:max_len], "truncated": len(f_ids) > max_len}


def collate(examples: list[dict], pad_id: int) -> dict:
    n = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id)
    labels = torch.full((len(examples), n), -100)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}
