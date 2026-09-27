"""Builds the final fine-tuning sets (data/clean -> data/final).

Steps, in order:
  1. Load the cleaned datasets and the crisis safety dataset
  2. Exclude conversations flagged for crisis language unless they were
     reviewed with "decision": "keep" in data/review/crisis_flagged.jsonl
  3. EmpatheticDialogues: keep only conversations without self-disclosure
  4. Downsample Psych8k and EmpatheticDialogues (see MIX)
  5. Split train/val/test by conv_id (the same conversation or question never
     appears in two sets), deterministically from a hash of the conv_id
  6. Oversample AnnoMI and the crisis safety dataset (train only)
  7. Shuffle and write data/final/{train,val,test}.jsonl + mix_stats.json

The HOPE sample is copied as-is to data/final/hope_eval.jsonl (evaluation only).
"""
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict

from common import REPO_ROOT

CLEAN_DIR = REPO_ROOT / "data" / "clean"
REVIEW_DIR = REPO_ROOT / "data" / "review"
FINAL_DIR = REPO_ROOT / "data" / "final"
CRISIS_DATASET = REPO_ROOT / "data" / "crisis_safety_dataset.jsonl"

SEED = 42

# sample: max examples kept (None = all); oversample: repeats in train only
MIX = {
    "psych8k":       {"sample": 5000, "oversample": 1},
    "esconv":        {"sample": None, "oversample": 1},
    "amod":          {"sample": None, "oversample": 1},
    "annomi":        {"sample": None, "oversample": 3},
    "empathetic":    {"sample": 4000, "oversample": 1},
    "crisis_safety": {"sample": None, "oversample": 3},
}

SPLIT = {"train": 0.90, "val": 0.05, "test": 0.05}


# --- Loading ------------------------------------------------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_crisis_dataset():
    """Converts crisis_safety_dataset.jsonl to the unified format.
    Its single-turn examples reuse each user phrase with 2 different responses,
    so conv_id is derived from the first user message: both responses to the
    same phrase always land in the same split."""
    records = []
    for i, ex in enumerate(read_jsonl(CRISIS_DATASET)):
        first_user = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        key = hashlib.md5(re.sub(r"[^a-z0-9]", "", first_user.lower()).encode()).hexdigest()[:12]
        messages = [
            {**m, "train": True} if m["role"] == "assistant" else dict(m)
            for m in ex["messages"]
        ]
        records.append({
            "id": f"crisis_safety_{i}",
            "source": "crisis_safety",
            "conv_id": f"crisis_safety_{key}",
            "messages": messages,
            "meta": {"category": ex.get("category"), "type": ex.get("type")},
        })
    return records


def load_crisis_decisions():
    """ids reviewed with decision "keep"; anything else stays excluded."""
    path = REVIEW_DIR / "crisis_flagged.jsonl"
    if not path.exists():
        return set()
    return {r["id"] for r in read_jsonl(path) if r.get("decision") == "keep"}


# --- Split --------------------------------------------------------------------

def split_of(conv_id):
    """Deterministic split from a hash of conv_id: stable across runs and
    unaffected by adding or removing other conversations."""
    h = int(hashlib.md5(conv_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < SPLIT["train"]:
        return "train"
    if h < SPLIT["train"] + SPLIT["val"]:
        return "val"
    return "test"


# --- Stats --------------------------------------------------------------------

def trainable_words(record):
    return sum(len(m["content"].split()) for m in record["messages"]
               if m["role"] == "assistant" and m.get("train", True))


def main():
    rng = random.Random(SEED)
    keep_reviewed = load_crisis_decisions()
    stats = defaultdict(dict)
    pools = {}

    for source in MIX:
        records = load_crisis_dataset() if source == "crisis_safety" else read_jsonl(CLEAN_DIR / f"{source}.jsonl")
        stats[source]["available"] = len(records)

        before = len(records)
        records = [r for r in records
                   if not r["meta"].get("crisis_flag") or r["id"] in keep_reviewed]
        stats[source]["excluded: crisis not reviewed/kept"] = before - len(records)

        if source == "empathetic":
            before = len(records)
            records = [r for r in records if not r["meta"].get("self_disclosure")]
            stats[source]["excluded: self-disclosure"] = before - len(records)

        n = MIX[source]["sample"]
        if n is not None and len(records) > n:
            records = rng.sample(sorted(records, key=lambda r: r["id"]), n)
        stats[source]["selected"] = len(records)
        pools[source] = records

    splits = {"train": [], "val": [], "test": []}
    for source, records in pools.items():
        per_split = defaultdict(list)
        for r in records:
            per_split[split_of(r["conv_id"])].append(r)
        for name, recs in per_split.items():
            repeats = MIX[source]["oversample"] if name == "train" else 1
            splits[name].extend(recs * repeats)
            stats[source][f"{name} examples"] = len(recs) * repeats
        stats[source]["train trainable words"] = sum(trainable_words(r) for r in per_split["train"]) \
            * MIX[source]["oversample"]

    total_words = sum(s["train trainable words"] for s in stats.values())
    for source in MIX:
        stats[source]["train share of learning signal"] = \
            f"{100 * stats[source]['train trainable words'] / total_words:.1f}%"

    # sanity check: no conversation in more than one split
    ids = {name: {r["conv_id"] for r in recs} for name, recs in splits.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    for name, recs in splits.items():
        rng.shuffle(recs)
        write_jsonl(recs, FINAL_DIR / f"{name}.jsonl")
    shutil.copy(CLEAN_DIR / "hope_eval.jsonl", FINAL_DIR / "hope_eval.jsonl")

    totals = {name: len(recs) for name, recs in splits.items()}
    with open(FINAL_DIR / "mix_stats.json", "w", encoding="utf-8") as f:
        json.dump({"totals": totals, "per_source": stats}, f, indent=2)

    for source in MIX:
        print(f"\n[{source}]")
        for k, v in stats[source].items():
            print(f"  {k}: {v}")
    print(f"\nTotals: {totals}")
    print(f"Output: {FINAL_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
