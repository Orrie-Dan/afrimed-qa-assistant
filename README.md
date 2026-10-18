# AfriMed Assistant

**A medical question-answering assistant for African healthcare contexts.** It uses a small open LLM fine-tuned with LoRA on AfriMed-QA, behind an out-of-domain gate and a crisis-safety check.

| | |
|---|---|
| 🌐 **Live app** | https://huggingface.co/spaces/OrrieUzi/afrimed-assistant |
| 🎥 **Demo video** | https://youtu.be/r-1iQLhY5ek |
| 📄 **Report** | [`report/report.docx`](report/report.docx) (source: [`report/report.md`](report/report.md)) |

> ⚠️ **Not medical advice.** This is an academic project. The model can be wrong, sometimes confidently: see [Limitations](#limitations).

---

## What it does

A user types a health question. The app then works in three steps (`src/afrimed/assistant.py`):

1. **Crisis check.** A rule matches suicide and self-harm phrases and always shows crisis-support information, whatever happens next.
2. **Out-of-domain gate.** A TF-IDF + logistic-regression classifier decides whether the question is about human health. Non-medical questions are politely declined.
3. **Answer.** Medical questions go to **Qwen2.5-1.5B-Instruct + a LoRA adapter** fine-tuned on AfriMed-QA. The answer is streamed token by token and shown with a disclaimer.

## Key results

The test set has 806 MCQs and 246 SAQs, all from our own split of AfriMed-QA. Brackets are 95% bootstrap CIs. Full details: [`docs/results_run1.md`](docs/results_run1.md).

| Experiment | MCQ, option likelihood¹ | MCQ, generated answer² | SAQ ROUGE-L | SAQ BERTScore | Words per answer |
|---|---|---|---|---|---|
| E0 zero-shot Qwen2.5-0.5B (**baseline**) | 0.311 [0.281, 0.341] | 0.056 | 0.167 | 0.133 | 134 |
| E0 zero-shot Qwen2.5-1.5B | 0.479 [0.444, 0.512] | 0.360 | 0.177 | 0.160 | 136 |
| E1 LoRA 0.5B, SAQ+MCQ | 0.378 [0.344, 0.410] | 0.330 | 0.289 | 0.246 | 47 |
| E2 LoRA 0.5B, SAQ only | 0.319 [0.285, 0.352] | 0.030 | 0.331 | 0.314 | 36 |
| **E3 LoRA 1.5B, SAQ+MCQ (deployed)** | **0.508 [0.472, 0.542]** | **0.454** | 0.330 | **0.316** | 39 |

¹ The option letter the model rates most likely, on 744 single-answer MCQs. It doesn't depend on answer format.
² Exact-set match of the generated answer, on all 806 MCQs.

**Main findings**

The paired McNemar tests use Bonferroni correction ([`docs/results_run1.md`](docs/results_run1.md)).

- **Fine-tuning significantly improves how the models answer**, at both sizes.
  - At 1.5B the gain in option likelihood is *not* significant (p = 0.10). The improvement there is mostly that the model learned to commit to one option: multi-letter answers fell from 23% to 4%.
- **Model size is the strongest effect.** Zero-shot 1.5B beats fine-tuned 0.5B.
- **Training on MCQs matters.** The SAQ-only model (E2) loses MCQ ability.
- **Overlap metrics are not correctness.** Fine-tuning roughly doubles ROUGE-L, largely by learning the short, templated answer style of the crowdsourced data. Some high-ROUGE answers are factually wrong, so a manual correctness and safety rating is part of the evaluation ([`docs/rating_guide.md`](docs/rating_guide.md)).

**Out-of-domain gate** ([`docs/ood_gate.md`](docs/ood_gate.md)):

| Evaluation set | Balanced accuracy |
|---|---|
| Test set (7,210 questions from 8 real sources) | 0.984 |
| Held-out hand-written probe set (57 questions) | 0.84 |

The probe set was written to look for shortcuts, such as treating chit-chat or Africa-related questions as medical. The remaining weaknesses are animal-health questions and some non-clinical health questions. The crisis check exists because the gate misses some suicide-related questions.

**Deployment** (Hugging Face Spaces, free CPU tier): the median answer takes 15.8 s, with a range of 9–51 s ([`docs/benchmark_t4.md`](docs/benchmark_t4.md)).

## What was used and what was built

| Used (existing resources) | Built in this project |
|---|---|
| **AfriMed-QA v2.5** dataset (Olatunji et al., 2025) | A documented re-split of the answered questions (the official split puts all expert questions in test) |
| **Qwen2.5-0.5B/1.5B-Instruct** base models | LoRA fine-tuning and evaluation pipeline (`src/afrimed/`), with prompt format and loss masking |
| **Hugging Face Transformers, PEFT** (LoRA), PyTorch | A format-independent MCQ metric (option likelihood) and paired significance tests |
| **ROUGE-L** (`rouge-score`), **BERTScore** (`bert-score`) | The out-of-domain gate: data assembly from 8 sources, medical-content filtering, shortcut diagnosis, two probe sets |
| Gate negatives: AfriQA, TAT-QA, SQuAD v1.1, Dolly-15k, KisanVaani agriculture QA | Crisis-safety rule, streaming Gradio app, Space builder |
| scikit-learn, Gradio, Hugging Face Spaces | Blind manual-rating protocol, and all analysis in `docs/` |

## Repository layout

```
src/afrimed/        shared code: prompts, data, training, inference, metrics, evaluation, gate, assistant
scripts/            local CPU scripts: data download/prep, gate training, rating, notebook + Space builders
notebooks/          Colab notebooks (generated from src/ by scripts/build_*_notebook.py; don't hand-edit)
  runs/             executed copies of the notebooks, with their outputs (evidence for reported numbers)
app/app.py          Gradio app
data/processed/     our splits (QA and gate); raw files are downloaded, see below
data/ood_probe*.jsonl  hand-written gate probe sets (evaluation only)
models/gate.joblib  trained out-of-domain gate
results/            experiments table v2, significance tests, CPU latency; gate metrics;
                    test predictions (E0 1.5B, E3); manual-rating sheet
docs/               data findings, benchmarks, results, gate, deployment and rating guide
report/             final report (report.docx; report.md is its source) and figures
```

## Reproducing the results

**Local (CPU), Python 3.13:**

```bash
pip install -r requirements.txt
python -P scripts/download_data.py      # 34 raw files from pinned revisions, checksum-verified
python -P scripts/prepare_data.py       # AfriMed-QA -> data/processed/qa_*.jsonl, consumer_*.jsonl (seed 42)
python -P scripts/prepare_ood_data.py   # gate data -> data/processed/ood_*.jsonl
python -P scripts/train_gate.py         # gate -> models/gate.joblib, results/gate/
```

`prepare_data.py` checks the raw file's SHA-256 and produces byte-identical splits on every run. The Colab notebooks verify the same split hashes before training.

**Colab:** upload `data/processed/qa_{train,val,test}.jsonl` when asked, and keep results on Google Drive.

| Notebook | Runtime | What it does | Time |
|---|---|---|---|
| `01_t4_benchmark` | T4 GPU | memory/speed, used to choose the settings | ~25 min |
| `02_train_eval` | T4 GPU | **all experiments E0–E3**: training + val/test evaluation + experiments table | ~1.5–2 h |
| `03_mcq_likelihood` | T4 GPU | format-independent MCQ scores from the saved adapters | ~10 min |
| `04_significance` | CPU | paired McNemar tests | ~1 min |
| `05_cpu_latency` | CPU | app latency, used to choose the deployed model | ~20 min |

**Training settings:**
- LoRA r=16, alpha=32, dropout 0.05 on all attention and MLP projections.
- lr 2e-4 with cosine decay and 5% warm-up, effective batch size 16, 3 epochs.
- The best epoch is selected by val loss, in fp16, with max_len 512.
- Seed 42.

**App:**
1. Download the adapter (from the Space's `adapter/` folder, or from `runs/E3_lora_1.5B_saq+mcq/adapter` on Drive) into `models/adapter/`.
2. Run `python scripts/build_space.py`.
3. Run `python scripts/upload_space.py <user>/<space>`.

Details: [`docs/deploy.md`](docs/deploy.md).

## Limitations

- **Correctness.** The deployed model sometimes states wrong facts confidently. In the live test, it gave primaquine as the first-line treatment for uncomplicated malaria; artemisinin-based combination therapy is correct. It can also loop on list-style answers.
- **Evaluation.**
  - ROUGE-L and BERTScore measure overlap, not correctness.
  - Some reference answers in the dataset are wrong.
  - The manual rating is done by a single non-expert rater.
- **Data.** Most answered questions come from Nigeria, Kenya, South Africa and Ghana. Crowdsourced and expert questions differ strongly in style.
- **Gate.** It is lexical, so animal-health questions and some health questions without clinical vocabulary are misrouted.
- **Hosting.** The app runs on CPU, so answers take seconds to tens of seconds.

## Data and licences

| Resource | Licence |
|---|---|
| AfriMed-QA v2.5 (GitHub `intron-innovation/AfriMed-QA`) | **CC-BY-NC-SA-4.0** |
| AfriQA | CC-BY-4.0 |
| SQuAD v1.1 | CC-BY-SA-4.0 |
| Dolly-15k | CC-BY-SA-3.0 |
| TAT-QA | MIT (repository) |
| KisanVaani agriculture QA | Apache-2.0 |

The processed files in `data/processed/` are derived from these datasets and keep their licences. Because AfriMed-QA is NC-SA, this project is **non-commercial**.

## Citation of the main dataset

> Olatunji, T., et al. (2025). *AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset.* ACL 2025. arXiv:2411.15640.
