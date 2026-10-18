# Training and Evaluation: Run 1

- **Source:** `notebooks/runs/02_train_eval_run1.ipynb`
- **Environment:** Colab T4, torch 2.11.0, transformers 5.18.0, peft 0.21.1.
- **Settings:** test = 806 MCQ + 246 SAQ; val = 403 MCQ + 123 SAQ. Seed 42, greedy decoding.

## Experiments table (test; 95% bootstrap CI in brackets)

| Run | MCQ exact-set acc | Single | Multi | SAQ ROUGE-L | SAQ BERTScore | Words per SAQ answer | Val MCQ | Val ROUGE-L | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 zero-shot 0.5B | 0.056 [0.041, 0.072] | 0.059 | 0.016 | 0.167 [0.155, 0.179] | 0.133 [0.107, 0.158] | 134 | 0.087 | 0.165 | – | – |
| E0 zero-shot 1.5B | 0.360 [0.326, 0.392] | 0.376 | 0.161 | 0.177 [0.163, 0.190] | 0.160 [0.133, 0.185] | 136 | 0.382 | 0.168 | – | – |
| E1 LoRA 0.5B, SAQ+MCQ | 0.330 [0.298, 0.366] | 0.355 | 0.032 | 0.289 [0.266, 0.312] | 0.246 [0.204, 0.286] | 47 | 0.355 | 0.261 | 1 | 19.8 |
| E2 LoRA 0.5B, SAQ only | 0.030 [0.019, 0.042] | 0.027 | 0.065 | 0.331 [0.306, 0.353] | 0.314 [0.276, 0.349] | 36 | 0.052 | 0.289 | 2 | 4.9 |
| **E3 LoRA 1.5B, SAQ+MCQ** | **0.454 [0.419, 0.488]** | 0.481 | 0.129 | 0.330 [0.306, 0.355] | 0.316 [0.277, 0.351] | 39 | **0.511** | **0.298** | 1 | 34.5 |

**SAQ ROUGE-L by tier (test):**

| Run | Expert (n=72) | Crowdsourced (n=174) |
|---|---|---|
| E0 0.5B | 0.094 | 0.198 |
| E0 1.5B | 0.101 | 0.208 |
| E1 | 0.105 | 0.366 |
| E2 | 0.130 | 0.413 |
| E3 | 0.160 | 0.400 |

MCQ parse failures were 0–0.9% in every run, so the letter extraction is not what is driving the scores.

## Findings (confirmed from the outputs)

1. **E3 (1.5B + LoRA) is the best model on validation**, for both MCQ (0.511) and SAQ ROUGE-L (0.298). Choosing it uses val only; the test numbers above are reported, not used for selection.
2. **Fine-tuning helps both sizes on SAQ.** ROUGE-L rises from 0.17–0.18 to 0.29–0.33, and BERTScore from 0.13–0.16 to 0.25–0.32. The CIs do not overlap.
3. **On MCQ, fine-tuning helps at 1.5B**: 0.360 → 0.454, with non-overlapping CIs.
4. **Overfitting after epoch 1.**
   - For E1 and E3, val loss was lowest after epoch 1. Train loss kept falling, from about 1.0 to 0.45 for E1 and from 0.85 to 0.30 for E3, while val loss rose.
   - Keeping the best epoch handled this, but 3 epochs at lr 2e-4 is more than these models need.
5. **Training only on SAQ (E2) destroys MCQ ability** (0.030). It gives the best SAQ scores at 0.5B, but those gains are within the CIs of E3.
6. **The gains come almost entirely from crowdsourced SAQs**, which rise from 0.20 to about 0.40. Expert SAQs only rise from 0.09–0.10 to 0.11–0.16.
   - Crowdsourced answers follow a template ("X is a leading cause of Y in Africa"), and part of the gain is the model learning that template and the shorter length (134 → about 40 words).
7. **High ROUGE-L does not mean a correct answer.** Examples from E1's test predictions:
   - "Which parasitic infection is a leading cause of seizures in Africa?" The model answered "Schistosomiasis…" (ROUGE-L 0.74, but **wrong**; the reference says cerebral malaria). It copied the reference's sentence structure.
   - "Leading infectious cause of blindness in Africa?" The fine-tuned model said "Malaria" (wrong). The zero-shot model said "Trachoma", which is defensible.
   - "African race is a modifiable risk factor for hypertension. True or False?" The fine-tuned model said "True" (**wrong**). The zero-shot model said "False" (right).
   - Total gastrectomy question: the fine-tuned model degenerated into a repeated list ("Hypertriglyceridemia" over and over).

   This is the strongest evidence for not treating automatic overlap metrics as correctness.

## Confirmed: the low E0-0.5B and E2 MCQ scores come from answering several letters

This was checked against `predictions_test.jsonl` for both runs. 92% of test MCQs (744/806) have one correct option.

| Run | Answers that list >1 letter | Most common answers | Single-answer questions where the gold letter is among those listed | Exact |
|---|---|---|---|---|
| E0 zero-shot 0.5B | 75% (603/806) | "C,D" ×147, "E" ×125, "B,C" ×122 | 52.4% | 5.9% |
| E2 LoRA 0.5B, SAQ only | 87% (697/806) | "C,D" ×141, "A,B,C,D,E" ×108 | 64.0% | 2.7% |

- The prompt's "Select the correct option(s)" leads these models to hedge across several options, and exact-set scoring marks every hedge wrong.
- So **generative exact-set accuracy measures format-following as well as knowledge**. It is still the right metric for how the assistant behaves, but it is not a fair measure of what the zero-shot model knows.
- **Fix:** `notebooks/03_mcq_likelihood.ipynb` adds option-likelihood accuracy on single-answer questions: the argmax over A–E after "Answer:". It works for every run, reuses the saved adapters, and needs no retraining.
- Report both MCQ metrics in the experiments table and explain the difference.

## Option-likelihood results (notebook 03, `runs/experiments_table_v2.csv`)

Test set, 744 single-answer MCQs; 95% bootstrap CI in brackets.

| Run | Likelihood acc | Generative single acc | Multi-letter rate | Val likelihood |
|---|---|---|---|---|
| E0 zero-shot 0.5B | 0.311 [0.281, 0.341] | 0.059 | 74% | 0.339 |
| E0 zero-shot 1.5B | 0.479 [0.444, 0.512] | 0.376 | 23% | 0.519 |
| E1 LoRA 0.5B, SAQ+MCQ | 0.378 [0.344, 0.410] | 0.355 | 7% | 0.403 |
| E2 LoRA 0.5B, SAQ only | 0.319 [0.285, 0.352] | 0.027 | 85% | 0.325 |
| E3 LoRA 1.5B, SAQ+MCQ | 0.508 [0.472, 0.542] | 0.481 | 4% | 0.551 |

**Sanity checks: passed.**
- For E1 and E3 the likelihood and generative figures are within 3 points of each other, which is consistent with both measuring the same choice.
- E0 0.5B val (0.339 on 372 items) falls inside the uncertainty of the benchmark's 22/50 (44%; a 50-item sample has a CI of roughly ±14 points).

**What it changes:**
1. **Most of fine-tuning's MCQ gain is format: committing to one letter.**
   - 1.5B: generative accuracy rises 0.376 → 0.481, but likelihood only rises 0.479 → 0.508, and those CIs overlap.
   - 0.5B: likelihood rises 0.311 → 0.378, a gain of 6.7 points. The CIs only just separate.
   - The paired tests below settle this: the 0.5B gain is significant (p = 0.0006), the 1.5B gain is not (p = 0.10).
2. **SAQ-only training (E2) leaves MCQ knowledge unchanged** (0.319 vs 0.311). Its MCQ collapse is purely about format.
3. **Model size matters more than fine-tuning for medical knowledge.** Zero-shot 1.5B (0.479) beats fine-tuned 0.5B (0.378).
4. **E3 stays the best model on val** for both MCQ (0.551) and SAQ (0.298).
5. The "requires African expertise" slice has only about 49 single-answer test items, too few to draw conclusions from (E3: 0.612, E0 1.5B: 0.490).

## Paired significance tests (notebook 04, `runs/significance_tests.csv`)

- **Test:** exact two-sided McNemar on the test set.
- **Items:** likelihood uses 744 single-answer MCQs; generative uses exact-set accuracy on all 806 MCQs.
- **"Only A / only B":** questions where just one of the two models is correct.
- **Multiple comparisons:** there are 10 tests, so we use a Bonferroni threshold of **p < 0.005**. A p of 0.0 means p < 0.000005 after rounding.

| Comparison (A → B) | Metric | Acc A → B | Only A / only B | p | Significant? |
|---|---|---|---|---|---|
| Fine-tuning at 0.5B (E0 → E1) | likelihood | 0.311 → 0.378 | 78 / 128 | 0.0006 | **yes** |
| | generative | 0.056 → 0.330 | 16 / 237 | <1e-5 | **yes** |
| Fine-tuning at 1.5B (E0 → E3) | likelihood | 0.479 → 0.508 | 71 / 93 | 0.101 | **no** |
| | generative | 0.360 → 0.454 | 69 / 145 | <1e-5 | **yes** |
| Size, both fine-tuned (E1 → E3) | likelihood | 0.378 → 0.508 | 73 / 170 | <1e-5 | **yes** |
| | generative | 0.330 → 0.454 | 67 / 167 | <1e-5 | **yes** |
| Data mix (E1 SAQ+MCQ → E2 SAQ only) | likelihood | 0.378 → 0.319 | 125 / 81 | 0.0027 | **yes** (E1 better) |
| | generative | 0.330 → 0.030 | 255 / 13 | <1e-5 | **yes** (E1 better) |
| Size vs fine-tuning (E0 1.5B → E1 0.5B) | likelihood | 0.479 → 0.378 | 172 / 97 | 0.00001 | **yes** (1.5B zero-shot better) |
| | generative | 0.360 → 0.330 | 149 / 125 | 0.165 | **no** |

**Conclusions for the report:**
1. **Fine-tuning significantly improves how both models answer MCQs (generative).**
   - At 0.5B it also significantly improves which option the model prefers (likelihood).
   - At 1.5B the likelihood gain (+3.0 points) is **not significant** (p = 0.10). There, the generative gain comes mainly from committing to a single answer: the multi-letter rate fell from 23% to 4%.
2. **Model size is the largest and most reliable effect.** 1.5B beats 0.5B with fine-tuning on both metrics. The untuned 1.5B even beats the fine-tuned 0.5B on likelihood.
3. **Including MCQs in training matters for MCQ knowledge**, not only for format: E1 beats E2 on likelihood (p = 0.003).
   - E2 itself is no better than zero-shot (0.319 vs 0.311; this pair was not tested).
4. **Caveat:** the test only says whether two models differ. It says nothing about how correct either model's medical content is. SAQ correctness is addressed qualitatively in the report's error analysis (Section 7.2).

## Data issues found

- **Malformed MCQs:**
  - 1 test MCQ is an essay prompt with all options "N/A" and gold = all five.
  - 5 train MCQs have a gold answer that points at an "N/A" option.
  - The effect on the scores is negligible (1/806 on test), so this is documented, not re-split.
- 22% of MCQs have an "n/a" filler option. These are really 4-option questions, which is legitimate.
- 65 MCQs have options that start with their own letter ("A. Nevirapine…"). This is cosmetic.
