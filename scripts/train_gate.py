"""Train and evaluate the medical / non-medical gate (CPU, about a minute).

Usage: python -P scripts/train_gate.py   (after scripts/prepare_ood_data.py)

Model selection uses val only: feature set x C by val balanced accuracy at 0.5, then the decision
threshold that maximises val balanced accuracy. Test and the two hand-written probe sets are scored once
(data/ood_probe.jsonl = dev probe, used to diagnose v1; data/ood_probe_v2.jsonl = held-out probe).
Outputs: models/gate.joblib, results/gate/{metrics.json, test_errors.jsonl, probe_*_predictions.jsonl}
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from afrimed.gate import FEATURE_SETS, MedicalGate, build_pipeline  # noqa: E402

PROC, OUT = ROOT / "data" / "processed", ROOT / "results" / "gate"
C_GRID = (0.3, 1.0, 3.0, 10.0)
AFRICA = re.compile(r"\b(africa\w*|nigeria\w*|kenya\w*|ghana\w*|rwanda\w*|uganda\w*|tanzania\w*|malawi\w*|zambia\w*|"
                    r"zimbabwe\w*|ethiopia\w*|somalia|sudan\w*|egypt\w*|morocc\w*|senegal\w*|cameroon\w*|congo\w*|"
                    r"lagos|nairobi|kigali|accra|abuja|kampala|dar es salaam|johannesburg|cape town|"
                    r"yoruba|igbo|hausa|zulu|swahili|twi|bemba|kinyarwanda)\b", re.IGNORECASE)


def load(name):
    with open(name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scores(y, p, threshold):
    """Report the gate as a detector of OUT-of-domain questions (declined = positive class)."""
    y, p = np.asarray(y), np.asarray(p)
    pred_med = p >= threshold
    declined_true, declined_pred = (y == 0), ~pred_med
    prec, rec, f1, _ = precision_recall_fscore_support(declined_true, declined_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred_med.astype(int), labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "balanced_acc": round(balanced_accuracy_score(y, pred_med), 4),
        "ood_precision": round(prec, 4), "ood_recall": round(rec, 4), "ood_f1": round(f1, 4),
        "in_domain_recall": round(tp / max(tp + fn, 1), 4),
        "roc_auc": round(roc_auc_score(y, p), 4) if len(set(y)) > 1 else None,
        "ood_avg_precision": round(average_precision_score(declined_true, 1 - p), 4) if len(set(y)) > 1 else None,
        "confusion": {"ood_declined": int(tn), "ood_answered": int(fp), "med_declined": int(fn), "med_answered": int(tp)},
    }


def per_group_acc(rows, p, threshold, key):
    groups = defaultdict(list)
    for r, pi in zip(rows, p):
        groups[key(r)].append((pi >= threshold) == bool(r["label"]))
    return {g: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for g, v in sorted(groups.items())}


def main():
    train, val, test = (load(PROC / f"ood_{s}.jsonl") for s in ("train", "val", "test"))
    # probe_dev (v1) was used to diagnose the first gate's failures; probe_heldout (v2) was written before
    # the retrain and is scored only here, once.
    probes = {"probe_dev": load(ROOT / "data" / "ood_probe.jsonl"),
              "probe_heldout": load(ROOT / "data" / "ood_probe_v2.jsonl")}
    X = lambda rows: [r["text"] for r in rows]
    Y = lambda rows: [r["label"] for r in rows]

    # 1. Feature set x C on val (threshold 0.5).
    grid = []
    for feats in FEATURE_SETS:
        for C in C_GRID:
            pipe = build_pipeline(feats, C).fit(X(train), Y(train))
            pv = pipe.predict_proba(X(val))[:, 1]
            grid.append({"features": feats, "C": C, "val_balanced_acc": round(balanced_accuracy_score(Y(val), pv >= 0.5), 4),
                         "val_roc_auc": round(roc_auc_score(Y(val), pv), 4)})
            print(grid[-1], flush=True)
    best = max(grid, key=lambda g: (g["val_balanced_acc"], g["val_roc_auc"]))
    pipe = build_pipeline(best["features"], best["C"]).fit(X(train), Y(train))

    # 2. Threshold on val.
    pv = pipe.predict_proba(X(val))[:, 1]
    cands = np.round(np.arange(0.05, 0.96, 0.01), 2)
    threshold = float(max(cands, key=lambda t: (balanced_accuracy_score(Y(val), pv >= t), -abs(t - 0.5))))
    gate = MedicalGate(pipe, threshold, {"features": best["features"], "C": best["C"]})

    # 3. Score val/test/probe once.
    pt = gate.prob_medical(X(test))
    pp = {name: gate.prob_medical(X(rows)) for name, rows in probes.items()}
    africa_neg = [(r, p) for r, p in zip(test, pt) if r["label"] == 0 and AFRICA.search(r["text"])]
    metrics = {
        "selection_grid": grid, "selected": best, "threshold": threshold,
        "val": scores(Y(val), pv, threshold),
        "test": scores(Y(test), pt, threshold),
        "test_at_0.5": scores(Y(test), pt, 0.5),
        "test_by_source": per_group_acc(test, pt, threshold, lambda r: r["source"]),
        "test_africa_mentioning_negatives": {
            "n": len(africa_neg),
            "correctly_declined": round(sum(p < threshold for _, p in africa_neg) / max(len(africa_neg), 1), 4)},
    }
    for name, rows in probes.items():
        metrics[name] = scores(Y(rows), pp[name], threshold)
        metrics[f"{name}_by_category"] = per_group_acc(rows, pp[name], threshold, lambda r: r["category"])

    # Most informative word features (readable; char n-grams omitted).
    feats = pipe.named_steps["features"]
    names = feats.get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    word = [i for i, n in enumerate(names) if n.startswith("word__")]
    if word:
        order = sorted(word, key=lambda i: coef[i])
        metrics["top_words_non_medical"] = [names[i][6:] for i in order[:25]]
        metrics["top_words_medical"] = [names[i][6:] for i in order[::-1][:25]]

    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    gate.save(ROOT / "models" / "gate.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(OUT / "test_errors.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(test, pt):
            if (p >= threshold) != bool(r["label"]):
                f.write(json.dumps({**r, "prob_medical": round(p, 4)}, ensure_ascii=False) + "\n")
    for name, rows in probes.items():
        with open(OUT / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for r, p in zip(rows, pp[name]):
                f.write(json.dumps({**r, "prob_medical": round(p, 4), "correct": (p >= threshold) == bool(r["label"])},
                                   ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "selection_grid"}, indent=2))


if __name__ == "__main__":
    main()
