"""AnnoMI: Motivational Interviewing sessions, one row per utterance per annotator.

Raw: data/raw/AnnoMI-full.csv.
Issues handled: utterances repeated once per annotator (dedup on
transcript_id + utterance_id), low-quality MI sessions (deliberately bad
therapy, excluded), trailing spaces in topic, very long sessions (chunked).
"""
import pandas as pd

from common import RAW_DIR, build_messages, make_records, report, write_jsonl

SOURCE = "annomi"
ROLE_MAP = {"client": "user", "therapist": "assistant"}


def main():
    df = pd.read_csv(RAW_DIR / "AnnoMI-full.csv")
    stats = {"raw rows": len(df), "raw sessions": df["transcript_id"].nunique()}

    df = df.drop_duplicates(["transcript_id", "utterance_id"])
    stats["unique utterances"] = len(df)

    df = df[df["mi_quality"] == "high"]
    stats["high-quality sessions"] = df["transcript_id"].nunique()

    df["topic"] = df["topic"].str.strip()

    records = []
    for tid, session in df.sort_values(["transcript_id", "utterance_id"]).groupby("transcript_id"):
        turns = [(ROLE_MAP[r], t) for r, t in zip(session["interlocutor"], session["utterance_text"])]
        messages = build_messages(turns)
        meta = {"topic": session["topic"].iloc[0], "mi_quality": "high"}
        records.extend(make_records(SOURCE, f"{SOURCE}_{tid}", messages, meta))

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
