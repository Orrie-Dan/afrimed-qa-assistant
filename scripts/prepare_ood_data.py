"""Build train/val/test data for the medical vs non-medical question gate.

Usage:
    python -P scripts/prepare_ood_data.py        (after scripts/prepare_data.py)

Positives (label 1, in-domain medical), from our AfriMed-QA splits, so the gate split matches the QA split:
    consumer queries (consumer_*.jsonl) + MCQ/SAQ question stems (qa_*.jsonl).
Negatives (label 0, out of domain), all real questions:
    AfriQA (English translations of questions written by speakers of 8 African languages): Africa-related
        non-medical questions, so the gate cannot learn "mentions Africa => medical". Official splits are kept.
    TAT-QA: financial-report questions. Official train / dev / test_gold splits are kept.
    SQuAD v1.1: general Wikipedia questions. train.parquet is split into train/val by article, and
        validation.parquet is used as test, so no article appears in two splits.
    Dolly-15k: human-written instructions and questions (chat, how-to, creative, brainstorming), so the gate
        cannot learn "conversational / instruction phrasing => medical". Split 80/10/10 at random.
    KisanVaani agriculture QA: farmer questions on crops, livestock and pests, so animal and plant disease
        is learned as out of scope. Deduplicated, split 80/10/10 at random. Exempt from the medical-term filter
        (its "disease", "vaccine", "treat" are about animals and plants, which is the point).
Medical content inside the negative sources is removed (SQuAD articles listed in MEDICAL_SQUAD_TITLES,
plus any AfriQA/SQuAD/TAT-QA question matching MEDICAL_TERMS), so it is not mislabelled as out of domain.
What was removed is written to data/processed/ood_removed_negatives.jsonl for review.
"""
import glob
import json
import random
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ood"
PROC = ROOT / "data" / "processed"
SEED = 42
# Negatives kept per split for the large sources (AfriQA is used in full).
CAP = {"train": 4000, "val": 800, "test": 800}

MEDICAL_SQUAD_TITLES = {
    "Antibiotics", "Asthma", "Bacteria", "Black_Death", "Brain", "Circadian_rhythm", "Diarrhea", "Digestion",
    "Immune_system", "Immunology", "Infection", "Myocardial_infarction", "Nutrition", "Pain",
    "Pharmaceutical_industry", "Pharmacy", "Tuberculosis", "Gene", "Genome", "Memory", "Emotion",
    "Adolescence", "Sexual_orientation", "Bill_%26_Melinda_Gates_Foundation", "United_Nations_Population_Fund",
}
MEDICAL_TERMS = re.compile(r"\b(" + "|".join([
    r"health\w*", r"disease\w*", r"illness\w*", r"sick\w*", r"medic\w*", r"doctor\w*", r"nurs\w*", r"hospital\w*",
    r"clinic\w*", r"patient\w*", r"symptom\w*", r"treat(?:ment|ed|ing|s)?", r"cure\w*", r"drug\w*", r"vaccin\w*",
    r"virus\w*", r"viral", r"infect\w*", r"epidemic\w*", r"pandemic\w*", r"covid\w*", r"corona\w*", r"ebola",
    r"malaria", r"hiv", r"aids", r"tubercul\w*", r"cholera", r"cancer\w*", r"diabet\w*", r"mortality", r"death rate",
    r"life expectancy", r"pregnan\w*", r"birth\w*", r"maternal", r"blood", r"heart", r"surg\w*", r"pharma\w*",
    r"mental", r"psychiatr\w*", r"injur\w*", r"wound\w*", r"pain", r"fever", r"nutrition\w*", r"malnutrition",
    r"anaemi\w*", r"anemi\w*", r"organ", r"body", r"cell", r"brain", r"lung\w*", r"kidney\w*", r"liver",
]) + r")\b", re.IGNORECASE)


# Agriculture questions skip MEDICAL_TERMS, but the few about human health are still dropped.
AGRI_HUMAN_HEALTH = re.compile(r"\b(cancer\w*|digest\w*|poison\w*|human health|pregnan\w*|hospital\w*|doctor\w*|"
                               r"death rates?|disease\w* in (?:humans|people))\b", re.IGNORECASE)


def is_medical(text: str) -> bool:
    return bool(MEDICAL_TERMS.search(text))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def positives(split: str) -> list[dict]:
    out = [{"text": r["text"], "label": 1, "source": "afrimed_consumer"}
           for r in load_jsonl(PROC / f"consumer_{split}.jsonl")]
    out += [{"text": r["question"], "label": 1, "source": f"afrimed_{r['question_type']}"}
            for r in load_jsonl(PROC / f"qa_{split}.jsonl")]
    return out


def afriqa() -> dict[str, list[str]]:
    by_split = {"train": [], "val": [], "test": []}
    split_map = {"train": "train", "dev": "val", "test": "test"}
    for f in sorted(glob.glob(str(RAW / "afriqa" / "*.json"))):
        split = split_map[Path(f).stem.split(".")[1]]
        by_split[split] += [r["translated_question"] for r in load_jsonl(f) if r.get("translated_question")]
    return by_split


def tatqa() -> dict[str, list[str]]:
    files = {"train": "tatqa_dataset_train.json", "val": "tatqa_dataset_dev.json", "test": "tatqa_dataset_test_gold.json"}
    out = {}
    for split, name in files.items():
        docs = json.load(open(RAW / "tatqa" / name, encoding="utf-8"))
        out[split] = [q["question"] for d in docs for q in d["questions"]]
    return out


def random_split(items: list[str], fracs=(0.8, 0.1)) -> dict[str, list[str]]:
    seen, uniq = set(), []
    for q in items:
        n = norm(q)
        if n and n not in seen:
            seen.add(n)
            uniq.append(q)
    random.Random(SEED).shuffle(uniq)
    a, b = int(len(uniq) * fracs[0]), int(len(uniq) * (fracs[0] + fracs[1]))
    return {"train": uniq[:a], "val": uniq[a:b], "test": uniq[b:]}


def dolly() -> dict[str, list[str]]:
    return random_split([r["instruction"] for r in load_jsonl(RAW / "dolly" / "databricks-dolly-15k.jsonl")])


def agri() -> dict[str, list[str]]:
    return random_split(pd.read_parquet(RAW / "agri" / "train.parquet").question.dropna().tolist())


def squad() -> tuple[dict[str, list[str]], list[dict]]:
    train = pd.read_parquet(RAW / "squad" / "train.parquet")
    test = pd.read_parquet(RAW / "squad" / "validation.parquet")
    removed = [{"text": q, "source": "squad", "reason": f"medical article: {t}"}
               for df in (train, test) for t, q in zip(df.title, df.question) if t in MEDICAL_SQUAD_TITLES]
    train, test = train[~train.title.isin(MEDICAL_SQUAD_TITLES)], test[~test.title.isin(MEDICAL_SQUAD_TITLES)]
    titles = sorted(train.title.unique())
    random.Random(SEED).shuffle(titles)
    val_titles = set(titles[: len(titles) // 10])
    out = {"train": train[~train.title.isin(val_titles)].question.tolist(),
           "val": train[train.title.isin(val_titles)].question.tolist(),
           "test": test.question.tolist()}
    return out, removed


def main() -> None:
    sources = {"afriqa": afriqa(), "tatqa": tatqa(), "dolly": dolly(), "agri": agri()}
    sources["squad"], removed = squad()
    no_term_filter = {"agri"}
    rng = random.Random(SEED)
    seen_neg = set()   # a negative question text appears in one split only
    stats = {}
    for split in ("train", "val", "test"):
        rows = positives(split)
        pos_norm = {norm(r["text"]) for r in rows}
        for name, by_split in sources.items():
            kept = []
            for q in by_split[split]:
                q = re.sub(r"\s+", " ", q).strip()
                n = norm(q)
                if not n or n in seen_neg or n in pos_norm:
                    continue
                if (AGRI_HUMAN_HEALTH.search(q) if name in no_term_filter else is_medical(q)):
                    removed.append({"text": q, "source": name, "reason": "medical term"})
                    continue
                seen_neg.add(n)
                kept.append(q)
            if name not in ("afriqa", "agri") and len(kept) > CAP[split]:
                kept = rng.sample(kept, CAP[split])
            rows += [{"text": q, "label": 0, "source": name} for q in kept]
        rng.shuffle(rows)
        with open(PROC / f"ood_{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[split] = pd.Series([r["source"] for r in rows]).value_counts().to_dict()
        stats[split]["total_pos"] = sum(r["label"] for r in rows)
        stats[split]["total_neg"] = sum(1 - r["label"] for r in rows)

    with open(PROC / "ood_removed_negatives.jsonl", "w", encoding="utf-8") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["removed_negatives"] = pd.Series([f"{r['source']}|{r['reason'].split(':')[0]}" for r in removed]).value_counts().to_dict()
    (PROC / "ood_split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
