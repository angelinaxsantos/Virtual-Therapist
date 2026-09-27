"""HOPE (public sample, 8 sessions): transcribed counseling sessions.

Raw: data/raw/hope/<session_id>.csv with columns ID, Type (T/P), Utterance, Dialog_Act.
With only 8 of 212 sessions public, this is written as an EVALUATION set
(data/processed/hope_eval.jsonl), not training data. When the full dataset
arrives, the same converter applies unchanged.
"""
import pandas as pd

from common import RAW_DIR, build_messages, make_records, report, write_jsonl

SOURCE = "hope"
ROLE_MAP = {"P": "user", "T": "assistant"}


def main():
    files = sorted((RAW_DIR / "hope").glob("*.csv"), key=lambda p: int(p.stem))
    stats = {"raw sessions": len(files)}

    records, n_utt = [], 0
    for path in files:
        df = pd.read_csv(path)
        n_utt += len(df)
        df["Type"] = df["Type"].str.strip()
        turns = [(ROLE_MAP[t], u) for t, u in zip(df["Type"], df["Utterance"]) if t in ROLE_MAP]
        messages = build_messages(turns)
        meta = {"session_id": path.stem, "dialog_acts": df["Dialog_Act"].fillna("").tolist()}
        records.extend(make_records(SOURCE, f"{SOURCE}_{path.stem}", messages, meta))
    stats["raw utterances"] = n_utt

    out = write_jsonl(records, f"{SOURCE}_eval.jsonl")
    report(SOURCE, stats, records, out)


if __name__ == "__main__":
    main()
