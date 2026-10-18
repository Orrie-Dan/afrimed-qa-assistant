"""Score the filled-in manual rating sheet (see docs/rating_guide.md).

Usage: python -P scripts/score_ratings.py
Reads results/rating/rating_sheet.csv + rating_key.csv, writes results/rating/rating_results.json.
With two systems it also compares them per question: a paired sign test on correctness, using only the
questions where they differ.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from afrimed.metrics import bootstrap_ci, mcnemar_exact  # noqa: E402

RATING = ROOT / "results" / "rating"


def read(name):
    with open(RATING / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def yes(v):
    return v.strip().lower() in ("yes", "y", "1", "true")


def main():
    sheet, key = read("rating_sheet.csv"), read("rating_key.csv")
    k = {(r["item"], r["answer_label"]): r for r in key}
    rows, missing = [], []
    for r in sheet:
        c = r["correctness_0_1_2"].strip()
        if c not in ("0", "1", "2"):
            missing.append((r["item"], r["answer_label"]))
            continue
        meta = k[(r["item"], r["answer_label"])]
        rows.append({**meta, "correctness": int(c), "unsafe": yes(r["unsafe_yes_no"]),
                     "reference_wrong": yes(r["reference_wrong_yes_no"]), "notes": r["notes"]})
    if missing:
        print(f"WARNING: {len(missing)} answers not rated yet (item, label): {missing[:10]}")

    def summary(items):
        n = len(items)
        return {"n": n,
                "mean_correctness_0_2": round(sum(i["correctness"] for i in items) / n, 3),
                "pct_correct_2": round(sum(i["correctness"] == 2 for i in items) / n, 3),
                "pct_partly_1": round(sum(i["correctness"] == 1 for i in items) / n, 3),
                "pct_wrong_0": round(sum(i["correctness"] == 0 for i in items) / n, 3),
                "pct_unsafe": round(sum(i["unsafe"] for i in items) / n, 3),
                "correct_ci95": bootstrap_ci([i["correctness"] == 2 for i in items])} if n else {"n": 0}

    by_sys = defaultdict(list)
    for r in rows:
        by_sys[r["system"]].append(r)
    out = {"by_system": {s: {"all": summary(v),
                             **{f"tier={t}": summary([i for i in v if i["tier"] == t]) for t in ("expert", "crowdsourced")}}
                         for s, v in by_sys.items()},
           "reference_flagged_wrong": sum(r["reference_wrong"] for r in rows if r["answer_label"] == "A")}

    if len(by_sys) == 2:
        a, b = sorted(by_sys)
        ca = {r["id"]: r["correctness"] == 2 for r in by_sys[a]}
        cb = {r["id"]: r["correctness"] == 2 for r in by_sys[b]}
        ids = sorted(ca.keys() & cb.keys())
        out["paired_correct_test"] = {"a": a, "b": b, **mcnemar_exact([ca[i] for i in ids], [cb[i] for i in ids])}

    RATING.mkdir(parents=True, exist_ok=True)
    (RATING / "rating_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
