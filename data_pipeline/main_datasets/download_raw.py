# data_pipeline/main_datasets/download_raw.py
import tarfile
import urllib.request
from pathlib import Path

from datasets import load_dataset

# Always resolve to <repo>/data/raw, regardless of working directory
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Psych8k (requires HuggingFace login and approved access)
ds = load_dataset("EmoCareAI/Psych8k")
ds["train"].to_json(RAW / "psych8k.jsonl")
print(f"Psych8k: {len(ds['train'])} examples -> {RAW / 'psych8k.jsonl'}")

# EmpatheticDialogues
url = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
tgz = RAW / "empatheticdialogues.tar.gz"
urllib.request.urlretrieve(url, tgz)
with tarfile.open(tgz) as t:
    t.extractall(RAW, filter="data")
print(f"EmpatheticDialogues extracted -> {RAW / 'empatheticdialogues'}")