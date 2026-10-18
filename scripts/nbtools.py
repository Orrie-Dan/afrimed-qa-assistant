"""Shared pieces for the generated Colab notebooks: setup, Drive, embedded src/afrimed, imports, data."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "afrimed"
MODULES = ["__init__.py", "prompts.py", "data.py", "infer.py", "metrics.py", "train.py", "evaluate.py",
           "experiments.py", "gate.py", "assistant.py"]

SETUP = r"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q peft rouge-score bert-score
# Colab ships torchao 0.10; recent peft raises ImportError if an older torchao is present. We don't use it.
!pip uninstall -y -q torchao
"""

DRIVE = r"""
import os
USE_DRIVE = True   # False -> results in /content only (lost when the runtime ends)
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/afrimed-qa-assistant"
else:
    BASE = "/content/afrimed-qa-assistant"
DATA_DIR, OUT_ROOT = f"{BASE}/data", f"{BASE}/runs"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT_ROOT, exist_ok=True)
print(BASE)
"""

IMPORTS = r"""
import sys, gc, json, hashlib, platform, random, time, importlib, importlib.util
sys.path.insert(0, "/content/afrimed_src")
import numpy as np, pandas as pd, torch, transformers, peft
from packaging import version
from transformers import AutoTokenizer
import afrimed.prompts, afrimed.data, afrimed.infer, afrimed.metrics, afrimed.train, afrimed.evaluate, afrimed.experiments
for m in (afrimed.prompts, afrimed.data, afrimed.infer, afrimed.metrics, afrimed.train, afrimed.evaluate,
          afrimed.experiments):
    importlib.reload(m)   # pick up re-written modules without restarting
from afrimed.prompts import parse_mcq
from afrimed.data import load_jsonl, write_jsonl
from afrimed.train import TrainConfig, load_base, add_lora, train
from afrimed.evaluate import predict, score, mcq_option_likelihood
from afrimed.experiments import EXPERIMENTS

assert torch.cuda.is_available(), "No GPU - set Runtime > Change runtime type > T4 GPU"
assert importlib.util.find_spec("torchao") is None, "torchao still installed: Runtime > Restart session, then Run all"
DTYPE_KW = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
ENV = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "transformers": transformers.__version__, "peft": peft.__version__, "python": platform.python_version()}
print(ENV)

# Answer-parser self-checks (cheap; catches regressions before an hour of GPU time).
F = frozenset
for text, want in [("Answer: B", F("B")), ("Answer: B, D", F("BD")), ("**Answer:** C", F("C")),
                   ("The answer is (D).", F("D")), ("Answer: D. Simple goiter", F("D")),
                   ("Answer: A, C, and E", F("ACE")), ("B. Graves' disease", F("B")),
                   ("A 35-year-old woman ...", None), ("Vitamin A deficiency", None),
                   ("Answer: Both A and B", None), ("I think option C is right", F("C"))]:
    assert parse_mcq(text) == want, (text, parse_mcq(text), want)
print("parser self-checks passed")
"""

DATA = r"""
EXPECTED_SHA256 = {
    "qa_train.jsonl": "17c8393f77d4d5773869a9063638532e843a08a932017725618ac342962f828e",
    "qa_val.jsonl": "fdefd6ac7619b53b689096bd5d0ffa63ae62596223450cece4ed7992a07aa0a9",
    "qa_test.jsonl": "c3066fc32696745d7a16e597b4ba8d7686031121998c45087cc059aaed974b6f",
}
missing = [f for f in EXPECTED_SHA256 if not os.path.exists(f"{DATA_DIR}/{f}")]
if missing:
    from google.colab import files
    print("Upload:", missing)
    for name, content in files.upload().items():
        with open(f"{DATA_DIR}/{os.path.basename(name)}", "wb") as fh:
            fh.write(content)
for f, expected in EXPECTED_SHA256.items():
    digest = hashlib.sha256(open(f"{DATA_DIR}/{f}", "rb").read()).hexdigest()
    assert digest == expected, f"{f}: hash mismatch - not the prepare_data.py seed-42 split"
TRAIN, VAL, TEST = (load_jsonl(f"{DATA_DIR}/qa_{s}.jsonl") for s in ("train", "val", "test"))
print(len(TRAIN), "train /", len(VAL), "val /", len(TEST), "test - hashes OK")
"""


class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def add_drive_and_src(self):
        """Drive paths + %%writefile of src/afrimed. Enough for CPU-only notebooks."""
        self.md("## Storage (Google Drive)")
        self.code(DRIVE)
        self.md("## Project code (`src/afrimed/`)")
        self.code("!mkdir -p /content/afrimed_src/afrimed")   # %%writefile needs the directory to exist
        for name in MODULES:
            self.code(f"%%writefile /content/afrimed_src/afrimed/{name}\n" + (SRC / name).read_text(encoding="utf-8"))

    def add_common(self):
        self.code(SETUP)
        self.add_drive_and_src()
        self.md("## Imports, environment, self-checks")
        self.code(IMPORTS)
        self.md("## Data (our split from `scripts/prepare_data.py`, seed 42)")
        self.code(DATA)

    def save(self, path):
        nb = nbf.v4.new_notebook(cells=self.cells, metadata={
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        })
        nbf.validate(nb)
        nbf.write(nb, path)
        print("wrote", path, len(self.cells), "cells")
