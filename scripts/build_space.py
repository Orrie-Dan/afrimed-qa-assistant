"""Assemble the Hugging Face Space folder (space/) from the project.

Usage: python scripts/build_space.py [--adapter models/adapter]

Needs the chosen LoRA adapter folder (adapter_config.json, adapter_model.safetensors, tokenizer files),
downloaded from Drive: runs/<experiment>/adapter -> models/adapter. Upload space/ as a Gradio Space
(see docs/deploy.md). Rebuild after changing app/ or src/afrimed/.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"
# Only what the app imports; training/eval code (and its torch-on-GPU assumptions) stays out of the Space.
APP_MODULES = ["__init__.py", "prompts.py", "infer.py", "gate.py", "assistant.py"]

# transformers/peft pinned to the versions the adapter was trained and evaluated with (Colab, run 1);
# scikit-learn pinned to the version that pickled models/gate.joblib. Gradio comes from sdk_version.
REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
transformers==5.18.0
peft==0.21.1
scikit-learn==1.8.0
packaging
"""

README = """---
title: AfriMed Assistant
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
license: cc-by-nc-sa-4.0
pinned: false
---

# AfriMed Assistant

A medical question-answering assistant focused on healthcare in African contexts. It is an academic project and **not medical advice**.

- **Model:** {base} with a LoRA adapter fine-tuned on AfriMed-QA v2.5 (Olatunji et al., 2025, arXiv:2411.15640, CC-BY-NC-SA-4.0).
- **Pipeline:** a rule-based crisis check, then an out-of-domain gate (TF-IDF + logistic regression), then the fine-tuned model.
- **Hardware:** free CPU hardware, so answers take several seconds.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "models" / "adapter"))
    args = ap.parse_args()
    adapter = Path(args.adapter)
    cfg = adapter / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(f"No adapter at {adapter}: download runs/<experiment>/adapter from Drive first.")
    base = json.loads(cfg.read_text())["base_model_name_or_path"]

    if SPACE.exists():
        shutil.rmtree(SPACE)
    (SPACE / "src" / "afrimed").mkdir(parents=True)
    (SPACE / "models").mkdir()
    shutil.copy(ROOT / "app" / "app.py", SPACE / "app.py")
    for m in APP_MODULES:
        shutil.copy(ROOT / "src" / "afrimed" / m, SPACE / "src" / "afrimed" / m)
    shutil.copy(ROOT / "models" / "gate.joblib", SPACE / "models" / "gate.joblib")
    shutil.copytree(adapter, SPACE / "adapter")
    (SPACE / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (SPACE / "README.md").write_text(README.format(base=base), encoding="utf-8")
    size = sum(f.stat().st_size for f in SPACE.rglob("*") if f.is_file()) / 1e6
    print(f"built {SPACE} for base model {base} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
