"""
Shared infrastructure for generating the project's Colab notebooks.

Each scripts/build_nbXX.py defines its cells as a list of ("md"|"code", source)
tuples and calls write_notebook(). Regenerate any notebook with:

    python scripts/build_nb00.py   # etc.

Notebooks are plain nbformat-4 JSON, runnable on Colab and locally.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Standard cells shared by every notebook
# ---------------------------------------------------------------------------

# Cell 1 of every notebook: mount Drive, clone repo, install, set secrets.
SETUP_CELL = '''\
# === Setup: mount Drive, clone repo, install requirements, set API keys ===
import os, sys, json, gc
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    PROJECT_DIR = Path("/content/drive/MyDrive/tara_project")
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    REPO_DIR = PROJECT_DIR / "tara-misalignment-persona"
    if not REPO_DIR.exists():
        # TODO: replace with your fork's URL before first Colab run
        !git clone https://github.com/YOUR_USERNAME/tara-misalignment-persona.git "{REPO_DIR}"
    %cd {REPO_DIR}
    !pip install -q -r requirements.txt
    for key in ["OPENROUTER_API_KEY", "HF_TOKEN"]:
        try:
            os.environ[key] = userdata.get(key)
        except Exception:
            print(f"WARNING: Colab secret {key} not set — API/judge cells will fail")
    RESULTS_DIR = PROJECT_DIR / "results"
else:
    # Local run: assume we are inside the repo (or notebooks/)
    REPO_DIR = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
    from dotenv import load_dotenv
    load_dotenv(REPO_DIR / ".env")
    RESULTS_DIR = REPO_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_DIR))
HF_TOKEN = os.environ.get("HF_TOKEN")

import torch as t
print(f"Repo: {REPO_DIR}")
print(f"Results dir: {RESULTS_DIR}")
print(f"CUDA: {t.cuda.is_available()}", end="")
print(f" ({t.cuda.get_device_name(0)})" if t.cuda.is_available() else "")\
'''


def quick_resume_cell(result_files: list[str]) -> str:
    """A cell that reports which expected outputs already exist on Drive."""
    files_literal = json.dumps(result_files, indent=4)
    return f'''\
# === Quick resume: skip recomputation if results already saved to Drive ===
EXPECTED_RESULTS = {files_literal}
existing = {{f: (RESULTS_DIR / f).exists() for f in EXPECTED_RESULTS}}
for f, ok in existing.items():
    print(("FOUND   " if ok else "missing ") + f)
# Downstream cells check `existing[...]` and skip work that is already done.\
'''


def write_notebook(filename: str, cells: list[tuple[str, str]]) -> Path:
    """Write cells [("md"|"code", source), ...] to notebooks/<filename>."""
    nb_cells = []
    for kind, source in cells:
        if kind == "md":
            nb_cells.append(
                {"cell_type": "markdown", "metadata": {}, "source": source}
            )
        elif kind == "code":
            nb_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": source,
                }
            )
        else:
            raise ValueError(f"unknown cell kind {kind!r}")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
        },
        "cells": nb_cells,
    }
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NOTEBOOKS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"wrote {out_path} ({len(nb_cells)} cells)")
    return out_path
