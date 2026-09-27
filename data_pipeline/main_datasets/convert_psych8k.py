"""Psych8k: single-turn counseling Q&A (Alpaca format).

Raw: data/raw/psych8k.jsonl with fields instruction, input, output.
The instruction is identical for all rows and is dropped (the system prompt
is added at training time). Origin is mixed: real counseling audio
transcripts rewritten by GPT-4.
"""
import pandas as pd

from common import RAW_DIR, build_messages, make_records, report, write_jsonl

SOURCE = "psych8k"


def main():
    df = pd.read_json(RAW_DIR / "psych8k.jsonl", lines=True)
    stats = {"raw rows": len(df), "distinct instructions (dropped)": df["instruction"].nunique()}

    records = []
    for i, row in df.iterrows():
        messages = build_messages([("user", row["input"]), ("assistant", row["output"])])
        records.extend(make_records(SOURCE, f"{SOURCE}_{i}", messages, chunk=False))

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
