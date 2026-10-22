"""Amod (mental_health_counseling_conversations): single-turn Q&A.

Raw: data/raw/AMOD.csv with columns Context, Response.
     data/raw/counsel_chat.csv (nbertagnolli/counsel-chat, optional) with
     questionID, questionTitle, questionText, answerText, ...
Issues handled:
  - empty responses and exact duplicate pairs
  - the same question written with small punctuation/case differences
    (grouped under one conv_id, so it can't land in both train and test)
  - one question with up to 94 answers (capped per question)
  - bilingual answers: one therapist appends a Spanish translation of the
    question and answer after the English text; that trailing block is cut
  - missing question titles: Amod keeps only the question body, so many
    questions depend on a title that was dropped ("She's in her early 20s,
    and she has a toddler." -> the title says it is about a stepdaughter).
    The title is recovered from CounselChat and put before the body
    ("Title\\n\\nBody"), unless the body already contains it. Each answer is
    matched by its own text first (most reliable), then by the question body.
  - answers paired with the wrong question in Amod: the answer matches a
    CounselChat question whose body is different from the Amod Context. Amod
    also has the same answer under its real question, so the wrongly paired
    copy is dropped
The conv_id and the record ids are still computed as before (from the original
Amod Context and row order), so they and the train/val/test split do not
change when the titles are added.
"""
import re
from collections import defaultdict

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect_langs

from common import RAW_DIR, build_messages, make_records, normalize_text, report, write_jsonl

DetectorFactory.seed = 0

SOURCE = "amod"
MAX_RESPONSES_PER_CONTEXT = 3
SEED = 42
COUNSEL_CHAT = RAW_DIR / "counsel_chat.csv"
ANSWER_KEY_CHARS = 300  # answers are matched on the first 300 normalized characters

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


def load_counsel_chat():
    """Return (questions by id, ids by body key, id by answer key), or None if the file is missing."""
    if not COUNSEL_CHAT.exists():
        print(f"[{SOURCE}] {COUNSEL_CHAT} not found: question titles are not added")
        return None
    cc = pd.read_csv(COUNSEL_CHAT, dtype=str).fillna("")
    questions, by_body, by_answer = {}, defaultdict(set), {}
    for _, r in cc.iterrows():
        qid = r["questionID"]
        title, body = normalize_text(r["questionTitle"]), normalize_text(r["questionText"])
        questions.setdefault(qid, (title, body))
        by_body[question_key(body)].add(qid)
        by_answer[question_key(normalize_text(r["answerText"]))[:ANSWER_KEY_CHARS]] = qid
    return questions, by_body, by_answer


def with_title(title, body):
    if not title or question_key(title) in question_key(body):
        return body
    return f"{title}\n\n{body}"


def add_titles(df, stats):
    """User turn = CounselChat title + body. Adds the columns "User" ("Context" is
    left as in Amod) and "repaired" (answer paired with the wrong question)."""
    df["User"] = df["Context"]
    df["repaired"] = False
    cc = load_counsel_chat()
    if cc is None:
        return df
    questions, by_body, by_answer = cc
    found = {"answer": 0, "question": 0, "none": 0, "re-paired": 0, "title already in body": 0}
    users, repaired = [], []
    for context, response in zip(df["Context"], df["Response"]):
        qid = by_answer.get(question_key(response)[:ANSWER_KEY_CHARS])
        if qid:
            found["answer"] += 1
        else:
            ids = by_body.get(question_key(context), set())
            qid = next(iter(ids)) if len(ids) == 1 else None
            found["question" if qid else "none"] += 1
        if not qid:
            users.append(context)
            repaired.append(False)
            continue
        title, body = questions[qid]
        if question_key(body) != question_key(context):
            found["re-paired"] += 1  # the answer belongs to another question
            repaired.append(True)
        else:
            repaired.append(False)
            body = context  # same question: keep the Amod text as it is
            if question_key(title) in question_key(body):
                found["title already in body"] += 1
        users.append(with_title(title, body))
    df["User"] = users
    df["repaired"] = repaired
    stats["CounselChat match by answer"] = found["answer"]
    stats["CounselChat match by question"] = found["question"]
    stats["no CounselChat match (no title)"] = found["none"]
    stats["answers paired with the wrong question (dropped)"] = found["re-paired"]
    stats["title already in the question body"] = found["title already in body"]
    return df


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

    # after the cap, so the cap, the ids and the split stay as before
    df = add_titles(df, stats)

    # conv_id groups all answers to the same question -> train/test split by question
    question_ids = {q: i for i, q in enumerate(sorted(df["qkey"].unique()))}
    records = []
    for n, (_, row) in enumerate(df.iterrows()):
        if row["repaired"]:  # skipped inside the loop so n, and the ids, stay as before
            continue
        qid = question_ids[row["qkey"]]
        messages = build_messages([("user", row["User"]), ("assistant", row["Response"])])
        for r in make_records(SOURCE, f"{SOURCE}_q{qid}", messages, chunk=False):
            r["id"] = f"{SOURCE}_q{qid}_a{n}"
            records.append(r)

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
