"""ESConv: emotional support conversations (seeker / supporter).

Raw: data/raw/ESConv.json.
Issues handled: many dialogues start with a supporter greeting and/or end
with an unanswered seeker turn (trimmed), frequent consecutive turns by
the same speaker (merged), survey scores stored as strings.
Support strategies are kept in meta, aligned with assistant turns.
"""
import json

from common import RAW_DIR, build_messages, make_records, normalize_text, report, write_jsonl

SOURCE = "esconv"
ROLE_MAP = {"seeker": "user", "supporter": "assistant"}


def to_int(value):
    return int(value) if str(value).strip().isdigit() else None


def strategies_per_assistant_turn(dialog):
    """Mirror build_messages merging to collect the strategies of each final assistant turn."""
    groups, prev = [], None
    for turn in dialog:
        if not normalize_text(turn["content"]):
            continue
        if turn["speaker"] != prev:
            groups.append((turn["speaker"], []))
            prev = turn["speaker"]
        strategy = turn.get("annotation", {}).get("strategy")
        if strategy:
            groups[-1][1].append(strategy)
    while groups and groups[0][0] != "seeker":
        groups.pop(0)
    while groups and groups[-1][0] != "supporter":
        groups.pop()
    return [sorted(set(s)) for speaker, s in groups if speaker == "supporter"]


def main():
    with open(RAW_DIR / "ESConv.json", encoding="utf-8") as f:
        data = json.load(f)
    stats = {"raw dialogues": len(data),
             "start with supporter (trimmed)": sum(d["dialog"][0]["speaker"] == "supporter" for d in data),
             "end with seeker (trimmed)": sum(d["dialog"][-1]["speaker"] == "seeker" for d in data)}

    records = []
    for i, d in enumerate(data):
        messages = build_messages([(ROLE_MAP[t["speaker"]], t["content"]) for t in d["dialog"]])
        seeker = d.get("survey_score", {}).get("seeker", {})
        meta = {
            "emotion_type": d.get("emotion_type"),
            "problem_type": (d.get("problem_type") or "").strip().lower(),
            "situation": normalize_text(d.get("situation")),
            "empathy": to_int(seeker.get("empathy")),
            "initial_emotion_intensity": to_int(seeker.get("initial_emotion_intensity")),
            "final_emotion_intensity": to_int(seeker.get("final_emotion_intensity")),
        }
        conv_records = make_records(SOURCE, f"{SOURCE}_{i}", messages, meta)
        # attach strategies chunk by chunk
        strategies = strategies_per_assistant_turn(d["dialog"])
        pos = 0
        for r in conv_records:
            n_assistant = sum(m["role"] == "assistant" for m in r["messages"])
            r["meta"] = {**meta, "strategies": strategies[pos:pos + n_assistant]}
            pos += n_assistant
        records.extend(conv_records)

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
