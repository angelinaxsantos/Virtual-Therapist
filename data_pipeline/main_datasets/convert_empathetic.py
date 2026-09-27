"""EmpatheticDialogues: short empathetic conversations grounded in an emotion.

Raw: data/raw/empatheticdialogues/{train,valid,test}.csv.
Issues handled: pandas cannot parse the files (some lines have extra
fields), commas encoded as "_comma_", no explicit roles. The first speaker
(odd utterance_idx) describes the situation -> user; the listener -> assistant.
All three official splits are merged; our own split is done later by conv_id.
"""
import csv

from common import RAW_DIR, build_messages, make_records, report, write_jsonl

SOURCE = "empathetic"
SPLITS = ["train", "valid", "test"]


def read_split(path):
    """Manual parse: first 6 fields never contain commas, so split on ',' and
    keep fields by position."""
    rows, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 6:
                skipped += 1
                continue
            rows.append(dict(zip(header[:6], parts[:6])))
    return rows, skipped


def main():
    stats, convs = {}, {}
    for split in SPLITS:
        rows, skipped = read_split(RAW_DIR / "empatheticdialogues" / f"{split}.csv")
        stats[f"{split} rows (skipped malformed)"] = f"{len(rows)} ({skipped})"
        for r in rows:
            convs.setdefault(r["conv_id"], []).append(r)
    stats["raw conversations"] = len(convs)

    records = []
    for conv_id, rows in convs.items():
        rows.sort(key=lambda r: int(r["utterance_idx"]))
        turns = [("user" if int(r["utterance_idx"]) % 2 == 1 else "assistant",
                  r["utterance"].replace("_comma_", ","))
                 for r in rows]
        messages = build_messages(turns)
        meta = {"emotion": rows[0]["context"],
                "situation": rows[0]["prompt"].replace("_comma_", ",").strip()}
        safe_id = conv_id.replace("hit:", "").replace("_conv:", "_")
        records.extend(make_records(SOURCE, f"{SOURCE}_{safe_id}", messages, meta, chunk=False))

    path = write_jsonl(records, f"{SOURCE}.jsonl")
    report(SOURCE, stats, records, path)


if __name__ == "__main__":
    main()
