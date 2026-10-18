# T4 Benchmark: Results (step 2)

- **Source run:** `notebooks/runs/01_t4_benchmark_run1.ipynb`
- **Environment:** Colab, Tesla T4 (15.6 GB), torch 2.11.0, transformers 5.18.0, peft 0.21.1, Python 3.13, 2 CPUs.
- **Setup:** LoRA r=16 / alpha=32 on all attention and MLP projections, fp16 autocast, gradient checkpointing, AdamW at lr 2e-4. Each timing is 20 measured steps after 3 warm-up steps.

## Token lengths (train set, chat-formatted)

| Type | n | p50 | p95 | p99 | max | > 512 tokens |
|---|---|---|---|---|---|---|
| MCQ | 2,824 | 139 | 211 | 257 | 462 | 0% |
| SAQ | 861 | 108 | 292 | 440 | 1,213 | 0.58% |
| All | 3,685 | 134 | 218 | 338 | 1,213 | 0.14% (5 examples) |

## Training

| Model | max_len | Batch | s/step | Examples/s | Peak memory (GB) | Minutes per epoch |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 512 | 4 | 0.223 | 17.9 | 4.95 | 3.4 |
| Qwen2.5-0.5B | 512 | 8 | 0.400 | 20.0 | 9.65 | 3.1 |
| Qwen2.5-0.5B | 1024 | 2 | 0.208 | 9.6 | 3.04 | 6.4 |
| Qwen2.5-0.5B | 1024 | 4 | 0.212 | 18.9 | 4.94 | 3.3 |
| Qwen2.5-1.5B | 512 | 4 | 0.409 | 9.8 | 9.19 | 6.3 |
| Qwen2.5-1.5B | 512 | 8 | OOM | – | – | – |
| Qwen2.5-1.5B | 1024 | 2 | 0.263 | 7.6 | 6.30 | 8.1 |
| Qwen2.5-1.5B | 1024 | 4 | 0.424 | 9.4 | 9.19 | 6.5 |

- Trainable parameters: 8.8M for 0.5B and 18.5M for 1.5B.
- The 512 and 1024 rows with batch 4 are nearly identical. Both used the same seeded batches, and almost no example is longer than 512 tokens, so the length limit rarely takes effect.
- The first and last losses come from single random batches, so they are too noisy to show a training trend. They only confirm that training runs.
- Memory is dominated by the logits for Qwen's 151k-token vocabulary, which is why it grows with batch size × sequence length.

## Inference (GPU, base model, zero-shot)

| Model | MCQ s/question | MCQ sanity accuracy (50 single-answer val items) | SAQ s/answer | SAQ tokens/s | Average new tokens |
|---|---|---|---|---|---|
| 0.5B | 0.043 | 22/50 (44%) | 4.1 | 27.7 | 113 |
| 1.5B | 0.056 | 28/50 (56%) | 5.1 | 23.5 | 119 |

- With 5 options, chance accuracy is 20%. The 50-item figures are a sanity check only, **not** the baseline.
- Zero-shot short answers are long, list-style essays that almost hit the 128-token cap. The reference answers are about 29 words. A model this verbose will score poorly on ROUGE-L whether or not it is correct, which is part of what fine-tuning should fix.

## CPU inference

Not measured. The run stopped before the CPU cell finished, so this is still open. The notebook now prints timing for each prompt as it goes.

## Decisions

- **max_len = 512.** It covers 99.86% of the training examples; the 5 longer SAQ answers get truncated.
- **Micro-batch 4, with gradient accumulation to an effective batch of 16**, for both sizes:
  - For 0.5B, batch 8 doubles memory for only about 10% more throughput.
  - For 1.5B, batch 8 runs out of memory.
- **Both sizes are affordable.** One epoch takes about 3.4 min (0.5B) or 6.3 min (1.5B), so 3 epochs fit easily in one Colab session. The 0.5B-vs-1.5B experiment stays in the plan.
- **Choosing the size for the app waits on the CPU timing.** HF Spaces free hosting runs on CPU.

## CPU inference (notebook 05, app configuration)

- **Setup:** Colab, 2 vCPUs, 13.6 GB RAM. A GPU runtime was attached, but the model ran on CPU because `Assistant.load` never moves it to the GPU.
- **Model loading:** fp32, LoRA merged, as the app does it.
- **Questions:** 5 val SAQs and 3 user-style questions, with a 200-token answer cap.

| Model | Load time | Process RAM | Mean s/answer | Max s/answer | Tokens/s | Mean new tokens |
|---|---|---|---|---|---|---|
| E1 0.5B + LoRA | 17.8 s | 3.3 GB | 12.3 | 47.1 | 4.58 | 56 |
| E3 1.5B + LoRA | 50.9 s | 8.2 GB | 28.8 | 75.1 | 1.75 | 50 |

**Decision: deploy E3**, the best model on val with significantly better MCQ results (paired tests, `docs/results_run1.md`). Its mean latency is under the 30–40 s limit set beforehand. To handle the long tail:
- **Streaming:** the answer is shown token by token (`Assistant.stream`, about 1.75 tokens/s). The user sees text within a second or two, not after about 30 s.
- **Answer cap lowered from 200 to 128 tokens.** Fine-tuned answers average about 50 tokens and the reference answers about 29 words, so only runaway answers are cut. The worst case falls to about 128 / 1.75 ≈ 73 s.

**Not yet measured:** latency on the actual HF Spaces CPU, which may differ from Colab's. Re-check it after deployment and note it in the report.

## Live deployment check (2026-10-14)

- **Space:** https://huggingface.co/spaces/OrrieUzi/afrimed-assistant (HF Spaces cpu-basic, Gradio 5.9.1, E3 adapter, 128-token cap).
- **Method:** called through the public Gradio API (`/gradio_api/call/ask`).

**Routing:**
- "What are the symptoms of malaria in children?" was **answered** (P(medical) = 0.98, 9.8 s).
- "How do I send money with M-Pesa?" was **declined** (0.14).
- "How can I help a friend who talks about suicide?" got **crisis+declined** (0.03). The crisis message was shown, as designed.

**Latency of 7 medical questions:** 9.1, 9.8, 11.7, 15.8, 19.3, 50.8 and 51.4 s (median 15.8 s). The two 50 s answers are repetition loops that ran to the 128-token cap.

**Notable answers** (for the error analysis):
- **Repetition loops.** Pre-eclampsia warning signs: "• Severe headache • Severe abdominal pain • Severe headache • …". Sickle cell management: a 20+ item bullet list.
- **Dangerously wrong.** "Primaquine is the first-line treatment for uncomplicated malaria in Nigeria." The standard first-line treatment is artemisinin-based combination therapy (e.g. artemether-lumefantrine).
- **Overconfident.** "i have been coughing for 3 weeks now is it TB" got "Yes, you should go to the hospital for a chest x-ray and sputum test". The advice to get tested is reasonable, but the "Yes" is unwarranted.
- **Reasonable.** Malaria symptoms in children, hypertension treatment, and iron-rich foods in pregnancy.
