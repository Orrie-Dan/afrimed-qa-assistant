"""Build a blind manual-rating sheet for SAQ answers (step 7: correctness and safety).

Usage:
    python -P scripts/make_rating_sheet.py --system E3=<E3 predictions_test.jsonl> [--system E0=<E0 1.5B file>]

Samples N_PER_TIER expert + N_PER_TIER crowdsourced test SAQs (seed 42). With two systems, both answers
to each question appear in random order under the labels A/B, so the rater doesn't know which model wrote
which. Writes results/rating/rating_sheet.csv (to fill in; opens in Excel) and rating_key.csv (which
system is A/B; don't open it until rating is done). Guidelines: docs/rating_guide.md.
"""
import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rating"
N_PER_TIER = 20
SEED = 42


def load_saq(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows if r["question_type"] == "saq"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, help="NAME=path/to/predictions_test.jsonl")
    args = ap.parse_args()
    systems = dict(s.split("=", 1) for s in args.system)
    preds = {name: load_saq(path) for name, path in systems.items()}

    ids = sorted(set.intersection(*(set(p) for p in preds.values())))
    first = next(iter(preds.values()))
    assert len(ids) == len(first), "prediction files cover different test questions"
    rng = random.Random(SEED)
    sample = []
    for tier in ("expert", "crowdsourced"):
        pool = [i for i in ids if first[i]["tier"] == tier]
        sample += rng.sample(pool, N_PER_TIER)
    rng.shuffle(sample)

    OUT.mkdir(parents=True, exist_ok=True)
    labels = "AB"[: len(preds)] if len(preds) > 1 else "A"
    sheet, key = [], []
    for n, qid in enumerate(sample, 1):
        names = list(preds)
        rng.shuffle(names)                       # blind: random A/B order per question
        for label, name in zip(labels, names):
            r = preds[name][qid]
            sheet.append({"item": n, "answer_label": label, "question": r["question"], "reference": r["reference"],
                          "model_answer": r["prediction"], "correctness_0_1_2": "", "unsafe_yes_no": "",
                          "reference_wrong_yes_no": "", "notes": ""})
            key.append({"item": n, "answer_label": label, "system": name, "id": qid, "tier": r["tier"],
                        "rougeL": r.get("rougeL"), "bertscore_f1": r.get("bertscore_f1")})

    # utf-8-sig so Excel shows accented characters correctly.
    for name, rows in (("rating_sheet.csv", sheet), ("rating_key.csv", key)):
        with open(OUT / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(sample)} questions x {len(preds)} system(s) = {len(sheet)} answers -> {OUT / 'rating_sheet.csv'}")


if __name__ == "__main__":
    main()
