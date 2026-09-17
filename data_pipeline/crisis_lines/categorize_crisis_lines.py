#!/usr/bin/env python3
"""
Adds an "audience" field to each entry in crisis_lines_full.json, detecting
by keyword whether the line targets a specific audience (elderly,
children/youth, students, LGBT, veterans, etc.) or is general-purpose.

Runs AFTER extract_crisis_lines.py and apply_manual_fixes.py - it's an
additional layer, it doesn't replace the previous ones.
"""
import json
import re

AUDIENCE_PATTERNS = {
    "elderly": [r"\belderly\b", r"\bsenior[s]?\b", r"\bolder (people|adults|patients|persons)\b", r"\baged care\b"],
    "children_youth": [r"\bchild(ren)?\b", r"\bkids?\b", r"\byouth\b", r"\byoung people\b", r"\bteen(ager)?s?\b",
                        r"\bminors?\b"],
    "students": [r"\bstudents?\b"],
    "lgbt": [r"\blgbt\w*\b", r"\bgay\b", r"\btrans(gender)?\b"],
    "veterans": [r"\bveterans?\b"],
    "women": [r"\bwomen'?s?\b"],
    "men": [r"\bmen'?s?\b", r"\bmensline\b"],
    "victims_abuse": [r"\bvictims? of (violence|abuse)\b", r"\bdomestic violence\b"],
}


def classify_audience(text: str) -> list[str]:
    matches = []
    for audience, patterns in AUDIENCE_PATTERNS.items():
        if any(re.search(p, text, re.I) for p in patterns):
            matches.append(audience)
    return matches or ["general"]


def add_audience_tags(path_in: str, path_out: str):
    with open(path_in, encoding="utf-8") as f:
        data = json.load(f)

    counts = {}
    for country, entries in data["countries"].items():
        for entry in entries:
            audiences = classify_audience(entry["text"])
            entry["audience"] = audiences
            for a in audiences:
                counts[a] = counts.get(a, 0) + 1

    data["_meta"]["audience_tagging_note"] = (
        "'audience' field added by keyword detection on the original text. "
        "'general' means no specific-audience keyword was found (it does not "
        "necessarily mean the line excludes other audiences)."
    )

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Audience distribution found:")
    for a, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {a}: {n}")


if __name__ == "__main__":
    add_audience_tags("crisis_lines_full.json", "crisis_lines_full.json")
