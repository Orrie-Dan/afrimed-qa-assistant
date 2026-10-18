"""Run a model over a split and score it (MCQ exact-set accuracy; SAQ ROUGE-L + BERTScore)."""
import gc

import torch

from .infer import build_prompt, generate
from .metrics import bert_score_f1, bootstrap_ci, rouge_l, summarise
from .prompts import LETTERS, gold_letters, parse_mcq

MCQ_MAX_NEW = 48     # room for a zero-shot model that explains before giving the letter
SAQ_MAX_NEW = 256    # SAQ references: median ~29 words, a few expert essays far longer


def predict(model, tok, records: list[dict], batch_size: int = 16, log=print) -> list[dict]:
    rows = []
    for qtype, max_new in (("mcq", MCQ_MAX_NEW), ("saq", SAQ_MAX_NEW)):
        recs = [r for r in records if r["question_type"] == qtype]
        log(f"generating {len(recs)} {qtype} answers ...")
        bs = batch_size * 2 if qtype == "mcq" else batch_size
        preds = generate(model, tok, [build_prompt(tok, r) for r in recs], max_new, bs)
        for r, p in zip(recs, preds):
            row = {k: r[k] for k in ("id", "question_type", "tier", "specialty", "country",
                                     "requires_african_expertise", "multi_answer", "question")}
            row["prediction"] = p
            if qtype == "mcq":
                parsed = parse_mcq(p)
                row.update(options=r["options"], gold=sorted(gold_letters(r)),
                           parsed=sorted(parsed) if parsed else None,
                           correct=parsed == gold_letters(r))
            else:
                row["reference"] = r["answer"]
            rows.append(row)
    return rows


@torch.no_grad()
def mcq_option_likelihood(model, tok, records: list[dict], batch_size: int = 16) -> tuple[list[dict], dict]:
    """Format-independent MCQ score: the option letter the model rates most likely right after "Answer:".

    Generative exact-set scoring also measures whether a model follows the answer format (zero-shot models
    often list several letters); this isolates which option the model prefers. It picks one letter, so the
    headline is accuracy on single-answer questions.
    """
    letter_ids = [sorted({tok(f" {c}", add_special_tokens=False)["input_ids"][0],
                          tok(c, add_special_tokens=False)["input_ids"][0]}) for c in LETTERS]
    recs = [r for r in records if r["question_type"] == "mcq"]
    prompts = [build_prompt(tok, r) + "Answer:" for r in recs]
    model.eval()
    old_side, tok.padding_side = tok.padding_side, "left"
    rows = []
    try:
        for i in range(0, len(recs), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            # With left padding a plain forward needs explicit positions (generate() does this itself).
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
            for r, lg in zip(recs[i:i + batch_size], logits):
                scores = [max(lg[t].item() for t in ids) for ids in letter_ids]
                choice = LETTERS[max(range(len(LETTERS)), key=scores.__getitem__)]
                gold = gold_letters(r)
                rows.append({"id": r["id"], "tier": r["tier"], "multi_answer": r["multi_answer"],
                             "requires_african_expertise": r["requires_african_expertise"],
                             "choice": choice, "gold": sorted(gold), "choice_in_gold": choice in gold})
    finally:
        tok.padding_side = old_side

    def acc(items):
        return {"n": len(items),
                "acc": round(sum(r["choice_in_gold"] for r in items) / len(items), 4) if items else None}

    single = [r for r in rows if not r["multi_answer"]]
    summary = {
        "single": {**acc(single), "acc_ci95": bootstrap_ci([r["choice_in_gold"] for r in single])},
        "single_by": {
            **{f"tier={t}": acc([r for r in single if r["tier"] == t]) for t in ("expert", "crowdsourced")},
            "african_expertise": acc([r for r in single if r["requires_african_expertise"]]),
        },
        # For multi-answer questions: is the top choice at least one of the correct options?
        "multi_top_choice_in_gold": acc([r for r in rows if r["multi_answer"]]),
    }
    return rows, summary


def score(rows: list[dict], log=print) -> dict:
    """Adds ROUGE-L/BERTScore to SAQ rows in place and returns the summary. Free the LLM before calling."""
    saq = [r for r in rows if r["question_type"] == "saq"]
    for r, s in zip(saq, rouge_l([r["prediction"] for r in saq], [r["reference"] for r in saq])):
        r["rougeL"] = round(s, 4)
    gc.collect()
    torch.cuda.empty_cache()
    log("computing BERTScore ...")
    f1, err = bert_score_f1([r["prediction"] for r in saq], [r["reference"] for r in saq])
    if f1 is not None:
        for r, s in zip(saq, f1):
            r["bertscore_f1"] = round(s, 4)
    summary = summarise(rows)
    summary["bertscore_error"] = err
    return summary
