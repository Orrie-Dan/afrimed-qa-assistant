# Out-of-Domain Gate (step 6)

The gate decides whether a question is in scope. If it isn't, the app declines instead of answering.

## Scope policy

- **In scope:** questions seeking information about human health: disease, symptoms, treatment, medicines, prevention, nutrition for health, mental health, reproductive and maternal health, first aid, and public health.
- **Out of scope:** everything else, including **animal and plant health**, health-sector admin (prices, insurance sign-up, opening hours, careers), finance, general knowledge, chit-chat and instructions.

## Data (`scripts/prepare_ood_data.py`)

All the training data is real. Only the probe sets are hand-written, and they are used for evaluation only.

| Label | Source | Train | Val | Test | Licence |
|---|---|---|---|---|---|
| Medical (1) | AfriMed-QA consumer queries | 7,208 | 901 | 901 | CC-BY-NC-SA-4.0 |
| Medical (1) | AfriMed-QA MCQ + SAQ question stems | 3,685 | 526 | 1,052 | CC-BY-NC-SA-4.0 |
| Non-medical (0) | AfriQA (English translations of questions by speakers of 8 African languages) | 3,268 | 3,223 | 2,635 | CC-BY-4.0 (GitHub) |
| Non-medical (0) | TAT-QA financial-report questions | 4,000 | 800 | 800 | MIT (repo) |
| Non-medical (0) | SQuAD v1.1 Wikipedia questions | 4,000 | 800 | 800 | CC-BY-SA-4.0 |
| Non-medical (0) | Dolly-15k human-written instructions and questions | 4,000 | 800 | 800 | CC-BY-SA-3.0 |
| Non-medical (0) | KisanVaani agriculture QA (deduplicated) | 1,762 | 220 | 222 | Apache-2.0 |

**How the data was assembled:**
- **Positives** follow our AfriMed-QA split, so no question the assistant is tested on appears in the gate's training data.
- **Negatives** use each source's official splits where they exist. SQuAD is split by article; Dolly and the agriculture set are split at random after deduplication. No negative text appears in two splits.
- **Medical content was removed from the negative sources:**
  - 25 SQuAD articles that are about medicine (Tuberculosis, Pharmacy, Immune_system and others).
  - Any AfriQA, TAT-QA, SQuAD or Dolly question that matches a list of medical terms.
  - Agriculture questions about human health (7, such as "skin cancer among farmers").

  Everything removed is logged in `data/processed/ood_removed_negatives.jsonl`. The agriculture set is exempt from the general term list, because its "disease", "vaccine" and "treat" refer to animals and plants, which is exactly what the gate has to learn to decline.
- **Probe sets** (hand-written, evaluation only):
  - `data/ood_probe.jsonl` (62 items) is the **dev probe**. It was used to diagnose v1.
  - `data/ood_probe_v2.jsonl` (57 items) is the **held-out probe**. It was written before v2 was trained and scored once.

  Categories: plain, African-context and colloquial medical; health-adjacent; Africa-related non-medical; finance; agriculture and veterinary; health admin; general and chit-chat.

## Model (`src/afrimed/gate.py`, `scripts/train_gate.py`)

- **Model:** TF-IDF features (word 1–2-grams and/or character 2–5-grams) fed into logistic regression with balanced class weights.
- **Selection:** the feature set and regularisation strength C are chosen by val balanced accuracy. The decision threshold is then the one that maximises val balanced accuracy.
- **Result:** v2 uses word+char features, C = 3 and threshold 0.40.
- **Speed:** it trains in about a minute on a CPU, and the saved model is `models/gate.joblib`.

## Results

The "declined" class is treated as positive for precision and recall.

| | v1: AfriQA + TAT-QA + SQuAD negatives | **v2: + Dolly + agriculture** |
|---|---|---|
| Test balanced accuracy | 0.990 | 0.984 |
| Test: non-medical correctly declined (recall) | 0.992 | 0.985 |
| Test: medical correctly answered (in-domain recall) | 0.988 | 0.983 |
| Test: Africa-mentioning non-medical questions declined | 0.995 (n=581) | 0.990 (n=590) |
| Dev probe balanced accuracy (n=62) | 0.799 | 0.873 |
| **Held-out probe balanced accuracy (n=57)** | **0.668** | **0.839** |

The v1 and v2 test sets differ, because v2's test set adds the Dolly and agriculture negatives, so their test rows are not directly comparable. Both versions were scored on the same held-out probe.

### What the numbers mean

1. **The test set overstates real-world performance.** Both versions score about 0.98–0.99 on the test set, which comes from the same sources as training, but only 0.67–0.84 on hand-written questions phrased the way a real user might.
   - The probe sets are small (57–62 items), so their numbers are rough, roughly ±0.1. They are still the more honest estimate of what the app will see.
2. **v1 had learned a phrasing shortcut.**
   - Its most "medical" features were "my", "it", "you", "me" and "should"; its most "non-medical" features were "who", "when" and "how many".
   - So chit-chat and instructions ("Tell me a joke", "hello", "Write a Python function…") were treated as medical.
   - Adding real conversational and instruction-style negatives (Dolly) fixed most of this: general-category accuracy on the held-out probe went from 0.20 to 0.90.
3. **The Africa shortcut did not appear.** About 99% of non-medical test questions that mention Africa or African places are declined, and so are all held-out probe questions in the Africa-related non-medical category.
4. **Remaining failures (v2, held-out probe):**
   - **Animal-health questions with symptoms are still treated as medical**, for example "My goats are coughing and have a runny nose", "How often should I deworm my sheep?". Lexical features can't separate animal from human symptoms well.
   - **Some in-scope health questions are declined**, especially ones without clinical vocabulary:
     - "How can I help a friend who talks about suicide?" (p = 0.03)
     - "How many hours of sleep does a teenager need?"
     - "What does a high white blood cell count mean?"

   The suicide case is the most important: **declining a crisis question is a safety failure** (see the app safeguard below).

## Implications for the app

- **Crisis safeguard:** a rule-based check for self-harm and suicide terms must run **before** the gate, and must always show crisis-support information, whatever the gate decides. The gate must never be the only line of defence for these questions.
- **Decline message:** the message for declined questions should say that health questions can be rephrased, because the gate wrongly declines about 2% of medical test questions and more of the informal ones.
- **Report framing:** describe the gate as a lightweight first filter with known weaknesses, namely animal health and non-clinical wording, and not as a solved problem.
