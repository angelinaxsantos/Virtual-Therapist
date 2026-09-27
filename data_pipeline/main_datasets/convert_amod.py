"""Amod (mental_health_counseling_conversations): single-turn Q&A.

Raw: data/raw/AMOD.csv with columns Context, Response.
Issues handled:
  - empty responses and exact duplicate pairs
  - the same question written with small punctuation/case differences
    (grouped under one conv_id, so it can't land in both train and test)
  - one question with up to 94 answers (capped per question)
  - bilingual answers: one therapist appends a Spanish translation of the
    question and answer after the English text; that trailing block is cut
"""
import re

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs

from common import RAW_DIR, build_messages, make_records, normalize_text, report, write_jsonl

DetectorFactory.seed = 0

SOURCE = "amod"
MAX_RESPONSES_PER_CONTEXT = 3
SEED = 42

SPANISH_HINT = re.compile(r"\b(?:que|estoy|tengo|puedes|años|pero|muy|también|los|las|una)\b", re.I)
# Segment starts: after sentence punctuation, before ¿/¡, glued sentences, dash separators, newlines
SEGMENT_START = re.compile(r"(?<=[.!?])\s+|(?=[¿¡])|(?<=[a-z0-9][.!?])(?=[A-Z])|\n")
SEPARATOR = re.compile(r"-{5,}")


def is_spanish(text, min_prob=0.9):
    try:
        top = detect_langs(text)[0]
    except LangDetectException:
        return False
    return top.lang == "es" and top.prob >= min_prob


def strip_spanish_block(text):
    """Cut the trailing Spanish block. Only applied to answers with clear
    Spanish content, and only if everything after the cut is Spanish."""
    if len(SPANISH_HINT.findall(text)) < 5:
        return text
    sep = SEPARATOR.search(text)
    if sep:
        return text[:sep.start()].strip()
    starts = [0] + [m.end() for m in SEGMENT_START.finditer(text)]
    for pos, nxt in zip(starts, starts[1:] + [len(text)]):
        segment = text[pos:nxt]  # one sentence only, so neighbouring text can't sway detection
        if segment.startswith(("¿", "¡")) or (len(segment.split()) >= 4 and is_spanish(segment)):
            if is_spanish(text[pos:]):
                return text[:pos].strip()
    return text


def question_key(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main():
    df = pd.read_csv(RAW_DIR / "AMOD.csv")
    stats = {"raw rows": len(df)}

    df["Context"] = df["Context"].map(normalize_text)
    df["Response"] = df["Response"].map(normalize_text)

    before = df["Response"].copy()
    df["Response"] = df["Response"].map(strip_spanish_block)
    stats["answers with Spanish block cut"] = int((before != df["Response"]).sum())

    df = df[(df["Context"] != "") & (df["Response"] != "")]
    stats["after dropping empty"] = len(df)

    df["qkey"] = df["Context"].map(question_key)
    df["rkey"] = df["Response"].map(question_key)
    df = df.drop_duplicates(["qkey", "rkey"])
    stats["after dropping duplicate pairs"] = len(df)

    # shuffle, then keep the first N answers of each question
    df = df.sample(frac=1, random_state=SEED)
    df = df[df.groupby("qkey").cumcount() < MAX_RESPONSES_PER_CONTEXT]
    stats[f"after cap of {MAX_RESPONSES_PER_CONTEXT} responses per question"] = len(df)
    stats["unique questions"] = df["qkey"].nunique()

    # conv_id groups all answers to the same question -> train/test split by question
    question_ids = {q: i for i, q in enumerate(sorted(df["qkey"].unique()))}
    records = []
    for n, (_, row) in enumerate(df.iterrows()):
        qid = question_ids[row["qkey"]]
        messages = build_messages([("user", row["Context"]), ("assistant", row["Response"])])
        for r in make_records(SOURCE, f"{SOURCE}_q{qid}", messages, chunk=False):
            r["id"] = f"{SOURCE}_q{qid}_a{n}"
            records.append(r)

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
