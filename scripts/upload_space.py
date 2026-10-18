"""Upload space/ (built by scripts/build_space.py) to a Hugging Face Space.

Usage: python scripts/upload_space.py <your-username>/afrimed-assistant
Asks for a Hugging Face access token with write permission (input is hidden) unless you are already logged in.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

SPACE = Path(__file__).resolve().parent.parent / "space"


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit("Usage: python scripts/upload_space.py <your-username>/afrimed-assistant")
    if not (SPACE / "app.py").exists():
        raise SystemExit("space/ not found: run python scripts/build_space.py first")
    repo = sys.argv[1]
    api = HfApi()
    try:
        api.whoami()
    except Exception:
        login()
    api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(SPACE))
    print(f"Uploaded. Build progress and logs: https://huggingface.co/spaces/{repo}")


if __name__ == "__main__":
    main()
