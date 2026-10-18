"""Build the project's train/val/test splits from the raw AfriMed-QA v2.5 CSV.

Usage:
    python -P scripts/prepare_data.py

Inputs:  data/raw/afri_med_qa_15k_v2.5_phase_2_15275.csv
Outputs: data/processed/qa_{train,val,test}.jsonl        answered MCQ + SAQ (assistant)
         data/processed/consumer_{train,val,test}.jsonl  consumer queries (in-domain side of the OOD gate)
         data/processed/split_stats.json

Decisions (see docs/data_findings.md for the evidence behind each):
- The official split is not used: it puts every expert-written question in test,
  so train and test would come from different authors. We re-split instead.
- Answered pool = MCQ + SAQ. Rows flagged quality == False are dropped, and so are
  exact duplicates (after normalisation).
- QA split is 70/10/20, stratified by question_type x tier x multi-answer, seed 42.
- Consumer queries are split 80/10/10, grouped by normalised text so a duplicate
  question never appears on both sides.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "afri_med_qa_15k_v2.5_phase_2_15275.csv"
OUT = ROOT / "data" / "processed"
RAW_SHA256 = "422be9b69bf8870cbc40e58217a9f18cd47f640b96ab60226e6ec4a92256cbe3"
SEED = 42
QA_FRACS = {"train": 0.7, "val": 0.1, "test": 0.2}
CQ_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def clean(text) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.replace("�", " ")  # stray replacement chars from the source encoding
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def assign_splits(keys: pd.Series, fracs: dict, seed: int) -> pd.Series:
    """Shuffle each stratum and cut it by the given fractions."""
    out = pd.Series(index=keys.index, dtype=object)
    for _, idx in keys.groupby(keys).groups.items():
        idx = pd.Index(idx).to_series().sample(frac=1, random_state=seed).index
        n = len(idx)
        n_train = round(n * fracs["train"])
        n_val = round(n * fracs["val"])
        out[idx[:n_train]] = "train"
        out[idx[n_train:n_train + n_val]] = "val"
        out[idx[n_train + n_val:]] = "test"
    return out


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_qa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    qa = df[df.question_type.isin(["mcq", "saq"])].copy()
    dropped = {"quality_false": int((qa.quality == False).sum())}  # noqa: E712 (NaN must stay)
    qa = qa[qa.quality != False]  # noqa: E712
    qa["qn"] = qa.question_clean.map(normalise)
    dup = qa.duplicated("qn", keep="first")
    dropped["duplicate_question"] = int(dup.sum())
    qa = qa[~dup]

    recs = pd.DataFrame({
        "id": qa.sample_id,
        "question_type": qa.question_type,
        "tier": qa.tier,
        "specialty": qa.specialty.where(qa.specialty.notna(), None),
        "country": qa.country,
        "requires_african_expertise": qa.requires_african_expertise.astype(bool),
        "question": qa.question_clean.map(clean),
        "options": qa.answer_options.map(lambda s: json.loads(s) if isinstance(s, str) else None),
        "answer": [
            sorted(c.split(",")) if t == "mcq" else clean(r)
            for t, c, r in zip(qa.question_type, qa.correct_answer, qa.answer_rationale)
        ],
        "rationale": [clean(r) if t == "mcq" else None
                      for t, r in zip(qa.question_type, qa.answer_rationale)],
        "official_split": qa.split,
    })
    empty = recs.question.isna() | recs.answer.map(lambda a: not a)
    dropped["empty_question_or_answer"] = int(empty.sum())
    recs = recs[~empty]
    recs["multi_answer"] = recs.answer.map(lambda a: isinstance(a, list) and len(a) > 1)

    strata = recs.question_type + "|" + recs.tier + "|" + recs.multi_answer.astype(str)
    recs["split"] = assign_splits(strata, QA_FRACS, SEED)
    return recs, dropped


def build_consumer(df: pd.DataFrame) -> pd.DataFrame:
    cq = df[df.question_type == "consumer_queries"].copy()
    cq = cq[cq.quality != False]  # noqa: E712
    cq["text"] = cq.question_clean.map(clean)
    cq = cq[cq.text.notna()]
    cq["qn"] = cq.text.map(normalise)
    # Split distinct questions, then keep one row per question so duplicates can't leak.
    cq = cq.drop_duplicates("qn", keep="first")
    keys = pd.Series("all", index=cq.index)
    cq["split"] = assign_splits(keys, CQ_FRACS, SEED)
    return cq[["sample_id", "text", "country", "split"]].rename(columns={"sample_id": "id"})


def main() -> None:
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise SystemExit(f"Raw file hash mismatch: {digest}")

    df = pd.read_csv(RAW)
    OUT.mkdir(parents=True, exist_ok=True)

    qa, dropped = build_qa(df)
    cq = build_consumer(df)
    for name in QA_FRACS:
        write_jsonl(qa[qa.split == name].drop(columns="split"), OUT / f"qa_{name}.jsonl")
    for name in CQ_FRACS:
        write_jsonl(cq[cq.split == name].drop(columns="split"), OUT / f"consumer_{name}.jsonl")

    stats = {
        "raw_sha256": digest,
        "raw_rows": len(df),
        "qa_dropped": dropped,
        "qa_by_split_type_tier": {
            f"{s}|{t}|{tier}": int(n)
            for (s, t, tier), n in qa.groupby(["split", "question_type", "tier"]).size().items()
        },
        "qa_multi_answer_by_split": qa[qa.multi_answer].split.value_counts().to_dict(),
        "consumer_by_split": cq.split.value_counts().to_dict(),
        "consumer_dropped_duplicates_or_quality": int(
            (df.question_type == "consumer_queries").sum() - len(cq)),
    }
    (OUT / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
