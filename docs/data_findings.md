# AfriMed-QA: Data Findings (step 1)

## Source and licence

- **File:** `afri_med_qa_15k_v2.5_phase_2_15275.csv`, from GitHub `intron-innovation/AfriMed-QA`, commit `9bb0ffc` (2025-02-10).
- **SHA-256:** `422be9b6…cbe3`. `scripts/prepare_data.py` checks this hash before running.
- **Licence:** the GitHub repo's LICENSE is **CC-BY-NC-SA-4.0**, copied to `data/raw/LICENSE_upstream.txt`.
  - The Hugging Face card (`intronhealth/afrimedqa_v2`, v2.4) says CC-BY-SA-4.0, and is gated (auto-approval, login required).
  - We follow the strictest of these: non-commercial, share-alike, with attribution. The app and repo must state this.
- **Paper:** Olatunji et al., *AfriMed-QA*, arXiv:2411.15640 (ACL 2025).

## Contents (15,275 rows, 30 columns)

| Type | Rows | Answer field | Tier |
|---|---|---|---|
| Consumer queries | 10,000 | none (169 have a rationale) | all crowdsourced |
| MCQ | 4,039 | `correct_answer` (option keys), 5 options each | 3,910 expert / 129 crowdsourced |
| SAQ | 1,236 | `answer_rationale` (free text) | 359 expert / 877 crowdsourced |

- **Countries:** mostly NG (7,577), KE (2,476), ZA (2,444) and GH (1,572). Answered questions come from GH, KE, MW, NG, ZA and TZ.
- **Specialties:** 38 labels, led by obstetrics and gynaecology, general surgery and paediatrics. Almost every crowdsourced SAQ has no specialty label.
- **`requires_african_expertise`:** 1,203 rows. We keep it as a slice for analysis.

## Findings that shaped the split

1. **The official split separates questions by author.**
   - Official test holds every expert MCQ and every expert SAQ.
   - Official train holds only crowdsourced items: 129 MCQs and 877 SAQs.
   - Training on it would mean testing on a different author population. **We re-split.**
2. **315 MCQs (7.8%) have more than one correct option.** These are nearly all expert questions (310). They are kept and stratified across splits. MCQ results will be reported separately for single-answer and multi-answer questions.
3. **The two tiers write in different styles.**
   - Crowdsourced SAQs are generic and templated: "What is the primary…" opens 103 of them. Answers are about 29 words.
   - Expert SAQs are clinical. Their answers range from one word ("Alpha-fetoprotein") to 900-word essays.
   - Results will be reported per tier.
4. **Label noise exists.** Example: a 4-week-old with projectile non-bilious vomiting is labelled "Duodenal atresia". The textbook answer is pyloric stenosis. Some reference answers are wrong, so evaluation should judge correctness itself and not only agreement with the reference.
5. **Encoding artefacts.** Some expert answers contain U+FFFD characters, which `clean()` replaces with spaces.
6. **Duplicates.**
   - Answered pool: one duplicate MCQ pair, dropped.
   - Consumer queries: 764 rows fall in 225 groups of duplicates, and 73 of those groups span the official train and test sets. We deduplicate before splitting.
7. **`quality == False`:** this flag is rare. We drop the 11 answered rows and 477 consumer rows that carry it.
8. **Expert ratings (`rated_*`)** exist for only 65 rows. That is too few to use for training or evaluation.

## Our split (`data/processed/`, seed 42)

Answered QA, 70/10/20, stratified by type × tier × multi-answer:

| Split | MCQ expert | MCQ crowd | SAQ expert | SAQ crowd | Total |
|---|---|---|---|---|---|
| train | 2,736 | 88 | 251 | 610 | 3,685 |
| val | 391 | 12 | 36 | 87 | 526 |
| test | 781 | 25 | 72 | 174 | 1,052 |

Consumer queries (the in-domain side of the out-of-domain gate), after deduplication and quality filtering, 80/10/10: 7,208 / 901 / 901.

**Checks performed:**
- No ID overlap between any pair of QA splits.
- No normalised-text overlap between the consumer splits.
- Rerunning the script gives byte-identical output.

## Implications for later steps

- The **test set is small for some slices**. SAQ expert has 72 items and MCQ crowd has 25. Report confidence intervals, or at least counts, for each slice.
- **Training data is dominated by expert MCQs.** The short-answers-only vs short-answers-plus-MCQ experiment is still meaningful, but the training formats differ: the model outputs option letters for one and free text for the other.
- The **expert SAQ answers vary hugely in length**, which makes ROUGE-L noisy on them. That is another reason to look beyond overlap metrics.
