"""LoRA fine-tuning loop (fp16 autocast, for a T4). Settings come from docs/benchmark_t4.md."""
import math
import random
import time
from dataclasses import asdict, dataclass, field

import torch
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM

from .data import collate, encode


@dataclass
class TrainConfig:
    model_name: str
    data_mix: str = "saq+mcq"          # "saq+mcq" or "saq"
    max_len: int = 512
    micro_batch: int = 4
    grad_accum: int = 4                # effective batch 16
    epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def select_mix(records: list[dict], data_mix: str) -> list[dict]:
    types = {"saq+mcq": {"saq", "mcq"}, "saq": {"saq"}}[data_mix]
    return [r for r in records if r["question_type"] in types]


def load_base(name: str, dtype=torch.float16, dtype_kw: str = "dtype"):
    return AutoModelForCausalLM.from_pretrained(name, **{dtype_kw: dtype}).cuda()


def add_lora(model, cfg: TrainConfig):
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=cfg.target_modules))
    for p in model.parameters():          # fp32 adapters for stable fp16 mixed precision
        if p.requires_grad:
            p.data = p.data.float()
    return model


@torch.no_grad()
def eval_loss(model, encoded: list[dict], pad_id: int, batch_size: int) -> float:
    """Token-weighted mean loss over answer tokens."""
    model.eval()
    total, n_tok = 0.0, 0
    for i in range(0, len(encoded), batch_size):
        batch = {k: v.cuda() for k, v in collate(encoded[i:i + batch_size], pad_id).items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += loss.item() * n
        n_tok += n
    model.train()
    return total / max(n_tok, 1)


def train(model, tok, train_recs: list[dict], val_recs: list[dict], cfg: TrainConfig, adapter_dir: str,
          log=print) -> dict:
    """Train; saves the adapter with the best val loss to adapter_dir and restores it at the end."""
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    train_enc = [encode(tok, r, cfg.max_len) for r in select_mix(train_recs, cfg.data_mix)]
    val_enc = [encode(tok, r, cfg.max_len) for r in select_mix(val_recs, cfg.data_mix)]
    log(f"train examples {len(train_enc)} (truncated {sum(e['truncated'] for e in train_enc)}), "
        f"val examples {len(val_enc)}")

    steps_per_epoch = math.ceil(len(train_enc) / cfg.micro_batch / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))
    scaler = torch.amp.GradScaler("cuda")

    history = {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "train_loss": [], "val": []}
    best, step, t0 = float("inf"), 0, time.time()
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_enc)))
        rng.shuffle(order)
        batches = [order[i:i + cfg.micro_batch] for i in range(0, len(order), cfg.micro_batch)]
        running = []
        for b_idx, b in enumerate(batches):
            batch = {k: v.cuda() for k, v in collate([train_enc[i] for i in b], tok.pad_token_id).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(loss.item())
            if (b_idx + 1) % cfg.grad_accum == 0 or b_idx == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 25 == 0:
                    avg = sum(running) / len(running)
                    history["train_loss"].append({"step": step, "loss": round(avg, 4)})
                    log(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
                    running = []
        v = eval_loss(model, val_enc, tok.pad_token_id, cfg.micro_batch * 2)
        improved = v < best
        history["val"].append({"epoch": epoch, "step": step, "val_loss": round(v, 4), "best": improved})
        log(f"== epoch {epoch} val_loss {v:.4f}{' (best, saved)' if improved else ''}")
        if improved:
            best = v
            model.save_pretrained(adapter_dir)
            tok.save_pretrained(adapter_dir)

    best_epoch = min(history["val"], key=lambda h: h["val_loss"])["epoch"]
    if best_epoch != cfg.epochs:
        set_peft_model_state_dict(model, load_peft_weights(adapter_dir))
        log(f"restored best adapter from epoch {best_epoch}")
    history.update(best_epoch=best_epoch, best_val_loss=round(best, 4), train_minutes=round((time.time() - t0) / 60, 1))
    model.eval()
    return history
