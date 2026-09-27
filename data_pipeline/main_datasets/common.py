"""Shared helpers for converting raw datasets to the unified chat JSONL format.

Output record:
    {"id": str, "source": str, "conv_id": str,
     "messages": [{"role": "user"|"assistant", "content": str}, ...],
     "meta": {...}}

Structural rules enforced here (content filters live in clean.py):
    - strict user/assistant alternation (consecutive same-role turns merged)
    - starts with "user", ends with "assistant"
    - no system prompt (added at training time)
"""
import html
import json
import re
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Approximate budget per example (~1.3 tokens per word -> ~2000 tokens)
MAX_WORDS_PER_EXAMPLE = 1500


def normalize_text(text):
    """Unicode/HTML/whitespace normalization. Returns '' for missing values."""
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)  # also turns \xa0 into a space
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_messages(turns):
    """turns: list of (role, text). Merges consecutive same-role turns,
    drops empty turns, trims leading assistant and trailing user turns."""
    messages = []
    for role, text in turns:
        text = normalize_text(text)
        if not text:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    return messages


def _words(message):
    return len(message["content"].split())


def chunk_messages(messages, max_words=MAX_WORDS_PER_EXAMPLE):
    """Split a long conversation into consecutive, non-overlapping chunks.
    Each chunk starts with a user turn and ends with an assistant turn,
    and cuts happen only between (user, assistant) pairs."""
    pairs = [messages[i:i + 2] for i in range(0, len(messages), 2)]
    chunks, current, size = [], [], 0
    for pair in pairs:
        pair_size = sum(_words(m) for m in pair)
        if current and size + pair_size > max_words:
            chunks.append(current)
            current, size = [], 0
        current.extend(pair)
        size += pair_size
    if current:
        chunks.append(current)
    return chunks


def make_records(source, conv_id, messages, meta=None, chunk=True):
    """Turn one conversation into one or more records."""
    if len(messages) < 2:
        return []
    parts = chunk_messages(messages) if chunk else [messages]
    records = []
    for i, part in enumerate(parts):
        records.append({
            "id": f"{conv_id}_c{i}" if len(parts) > 1 else conv_id,
            "source": source,
            "conv_id": conv_id,
            "messages": part,
            "meta": meta or {},
        })
    return records


def write_jsonl(records, filename):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def report(source, stats, records, path):
    n_conv = len({r["conv_id"] for r in records})
    n_turns = sum(len(r["messages"]) for r in records)
    print(f"\n[{source}] -> {path.relative_to(REPO_ROOT)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  output: {len(records)} examples, {n_conv} conversations, {n_turns} messages")
