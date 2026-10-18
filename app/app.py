"""AfriMed Assistant: Gradio app (CPU).

Pipeline (src/afrimed/assistant.py): crisis check -> medical/non-medical gate -> fine-tuned Qwen2.5 + LoRA.
Runs from the project (python app/app.py) or from the Space folder built by scripts/build_space.py.
Settings (environment variables): BASE_MODEL, ADAPTER (local folder), GATE (path to gate.joblib).
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def first_existing(*paths):
    return next((p for p in paths if p.exists()), None)


# In the Space, src/ and models/ sit next to app.py; in the project they are one level up.
sys.path.insert(0, str(first_existing(HERE / "src", HERE.parent / "src")))

import gradio as gr  # noqa: E402

from afrimed.assistant import Assistant  # noqa: E402

ADAPTER = os.environ.get("ADAPTER") or str(first_existing(HERE / "adapter", HERE.parent / "models" / "adapter"))
# The adapter records which base model it was trained on, so the two can't be mismatched.
BASE_MODEL = os.environ.get("BASE_MODEL") or json.loads(
    (Path(ADAPTER) / "adapter_config.json").read_text())["base_model_name_or_path"]
GATE = os.environ.get("GATE") or str(first_existing(HERE / "models" / "gate.joblib",
                                                    HERE.parent / "models" / "gate.joblib"))

assistant = Assistant.load(BASE_MODEL, ADAPTER, GATE, threads=os.cpu_count())

INTRO = """
# AfriMed Assistant
Ask a question about **human health and medicine**, with a focus on healthcare in African contexts.
A small open model (Qwen2.5 + LoRA) was fine-tuned on **AfriMed-QA**, a set of medical questions written by
contributors from 16 countries, mostly Nigeria, Kenya, South Africa and Ghana.

> ⚠️ **Not medical advice.** Answers can be wrong. For diagnosis or treatment, see a qualified health
> professional. In an emergency, go to the nearest health facility.

Questions outside human health (finance, farming, general knowledge, chit-chat) are politely declined.
Answers appear word by word and take about half a minute on the free CPU hardware.
"""

FOOTER = """
<small>Model: Qwen2.5-Instruct + LoRA fine-tuned on AfriMed-QA v2.5 (Olatunji et al., 2025, arXiv:2411.15640;
CC-BY-NC-SA-4.0). Non-commercial academic project. Out-of-domain gate: TF-IDF + logistic regression.</small>
"""

EXAMPLES = [
    "What are the symptoms of malaria in children?",
    "How can sickle cell disease be managed?",
    "i have been coughing for 3 weeks now is it TB",
    "Which foods help prevent anaemia during pregnancy?",
    "How do I send money with M-Pesa?",
    "Tell me a joke.",
]


def ask(question):
    # Stream the answer as it is generated: on free CPU hardware a full answer takes ~30 s.
    for r in assistant.stream(question):
        p = "–" if r["prob_medical"] is None else f"{r['prob_medical']:.2f}"
        status = f"{r['seconds']:.1f} s" if r["done"] else "generating…"
        yield r["markdown"], f"route: {r['route']}   |   P(medical) = {p}   |   {status}"


with gr.Blocks(title="AfriMed Assistant") as demo:
    gr.Markdown(INTRO)
    question = gr.Textbox(label="Your health question", lines=3, placeholder="e.g. What causes high blood pressure?")
    button = gr.Button("Ask", variant="primary")
    answer = gr.Markdown()
    details = gr.Textbox(label="How the question was handled", interactive=False)
    gr.Examples(EXAMPLES, inputs=question)
    gr.Markdown(FOOTER)
    button.click(ask, question, [answer, details])
    question.submit(ask, question, [answer, details])

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
