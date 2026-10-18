# Deploying the App (Hugging Face Spaces, free CPU)

## 0. Choose the model

Run `notebooks/05_cpu_latency.ipynb` on a Colab **CPU** runtime and decide based on its timings:
- If **E3** (1.5B) answers in an acceptable time (roughly under 30–40 s), deploy E3, the best model on val.
- Otherwise deploy **E1** (0.5B) and state the trade-off in the report.

## 1. Get the adapter onto this machine

From Google Drive, download the folder `MyDrive/afrimed-qa-assistant/runs/<experiment>/adapter/` for the chosen experiment. Place it at `models/adapter/`.

It should contain:
- `adapter_config.json`
- `adapter_model.safetensors`
- the tokenizer files

## 2. Build the Space folder

```bash
python scripts/build_space.py
```

This creates `space/`, which contains:
- `app.py`
- the app's modules from `src/afrimed/`
- `models/gate.joblib`
- `adapter/`
- `requirements.txt`
- `README.md`, which holds the Space settings

## 3. Create the Space and upload

1. Create a free Hugging Face account and an access token with **write** permission (Settings → Access Tokens).
2. Upload with the script, which asks for the token with the input hidden:

```bash
pip install huggingface_hub
python scripts/upload_space.py <your-username>/afrimed-assistant
```

   Or do the same steps by hand in Python:

```python
from huggingface_hub import HfApi, login
login()                                   # paste the token when asked
api = HfApi()
repo = "<your-username>/afrimed-assistant"
api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path="space")
```

3. Open `https://huggingface.co/spaces/<your-username>/afrimed-assistant`. The first build installs the dependencies and downloads the base model, which takes several minutes.

## 4. If the build fails

The Space has not been tested yet, so these are the most likely issues:
- **Gradio / huggingface_hub version conflict:**
  - The README pins `sdk_version: 5.9.1`, while `transformers==5.18.0` needs a recent `huggingface_hub`.
  - If the log shows an import error from `gradio` or `huggingface_hub`, set `sdk_version` in `space/README.md` to the newest Gradio version offered in the Space settings, and upload again.
- **Python version:** `scikit-learn==1.8.0` needs a recent Python, and the README sets `python_version: "3.12"`.
- **scikit-learn version:** the gate was saved with scikit-learn 1.8.0, so keep that pin, or retrain the gate with `python -P scripts/train_gate.py` under the new version.
- **Out of memory:** the 1.5B model needs about 6–7 GB in fp32, which fits the free tier's 16 GB. If it is killed anyway, switch to the E1 adapter.

## 5. Link it

Put the Space URL in `README.md` and in the report, as the brief requires.
