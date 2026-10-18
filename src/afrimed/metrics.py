"""Evaluation metrics: MCQ exact-set accuracy, ROUGE-L, BERTScore, bootstrap CIs."""
import numpy as np


def bootstrap_ci(values, n: int = 1000, seed: int = 0) -> list[float] | None:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Paired comparison of two models on the same items (exact two-sided McNemar / binomial test).

    a, b: per-item correctness, aligned by item. Only discordant pairs carry information.
    """
    from math import comb
    assert len(a) == len(b)
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"n": len(a), "acc_a": round(sum(a) / len(a), 4), "acc_b": round(sum(b) / len(b), 4),
            "only_a_correct": only_a, "only_b_correct": only_b, "p_value": round(p, 5)}


def rouge_l(preds: list[str], refs: list[str]) -> list[float]:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]


def bert_score_f1(preds: list[str], refs: list[str]) -> tuple[list[float] | None, str | None]:
    """roberta-large BERTScore F1, rescaled with the package baseline. Returns (scores, error)."""
    try:
        from bert_score import score
        _, _, f1 = score([p or " " for p in preds], refs, lang="en", model_type="roberta-large",
                         rescale_with_baseline=True, batch_size=32, verbose=False)
        return f1.tolist(), None
    except Exception as e:  # keep the other metrics if BERTScore breaks on a library update
        return None, f"{type(e).__name__}: {e}"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    """rows: per-item predictions with question_type, tier, multi_answer, scores. Returns metrics + slices."""
    mcq = [r for r in rows if r["question_type"] == "mcq"]
    saq = [r for r in rows if r["question_type"] == "saq"]
    out = {"n_mcq": len(mcq), "n_saq": len(saq)}

    def mcq_block(items):
        return {"n": len(items), "acc": _mean([r["correct"] for r in items]),
                "parse_fail": _mean([r["parsed"] is None for r in items])}

    out["mcq"] = {**mcq_block(mcq), "acc_ci95": bootstrap_ci([r["correct"] for r in mcq])}
    out["mcq_by"] = {
        "single": mcq_block([r for r in mcq if not r["multi_answer"]]),
        "multi": mcq_block([r for r in mcq if r["multi_answer"]]),
        **{f"tier={t}": mcq_block([r for r in mcq if r["tier"] == t]) for t in ("expert", "crowdsourced")},
        "african_expertise": mcq_block([r for r in mcq if r["requires_african_expertise"]]),
    }

    def saq_block(items):
        return {"n": len(items),
                "rougeL": _mean([r["rougeL"] for r in items]),
                "bertscore_f1": _mean([r.get("bertscore_f1") for r in items]),
                "pred_words": _mean([len(r["prediction"].split()) for r in items]),
                "ref_words": _mean([len(r["reference"].split()) for r in items])}

    out["saq"] = {**saq_block(saq),
                  "rougeL_ci95": bootstrap_ci([r["rougeL"] for r in saq]),
                  "bertscore_ci95": bootstrap_ci([r.get("bertscore_f1") for r in saq])}
    out["saq_by"] = {
        **{f"tier={t}": saq_block([r for r in saq if r["tier"] == t]) for t in ("expert", "crowdsourced")},
        "african_expertise": saq_block([r for r in saq if r["requires_african_expertise"]]),
    }
    return out
