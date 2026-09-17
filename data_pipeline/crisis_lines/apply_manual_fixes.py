#!/usr/bin/env python3
"""
Manual fixes to apply after extract_crisis_lines.py, for the cases automatic
extraction can't resolve on its own (numbers with phone-keypad letters, or
text ambiguous between schedules and phone numbers).

Each fix documents the original text and the reasoning behind the correction.
"""
import json

MANUAL_FIXES = {
    ("Slovakia", "051 / 7731 000 – Linka dôvery (Prešov) (Monday: 7.00 – 15.30, Tuesday to Thursday: 7.00 – 15.00, Friday: 7.00 – 14.30)."):
        {
            "numbers_found": ["051", "7731 000"],
            "fix_note": "The real numbers are 051 and 7731 000; the other automatically extracted values "
                        "(15.30, 15.00, 14.30) were schedule times written with a period, not phone numbers.",
        },
    ("Philippines", "The National Center for Mental Health Crisis Hotline can be reached by calling 1553. Also at 0917 899 USAP (8727), 0966 351 4518 or 0908 639 2672."):
        {
            "numbers_found": ["1553", "0917 899 8727", "0966 351 4518", "0908 639 2672"],
            "fix_note": "'USAP' is a phone-keypad mnemonic for '8727' (Wikipedia itself already gave that in "
                        "parentheses); the full number is 0917 899 8727.",
        },
    ("Philippines", "The Natasha Goulbourn Foundation provides 24/7 assistance to those who call (02) 8804–HOPE (4673)."):
        {
            "numbers_found": ["(02) 8804-4673"],
            "fix_note": "'HOPE' is a phone-keypad mnemonic for '4673' (confirmed by Wikipedia's own "
                        "parenthetical); full number (02) 8804-4673.",
        },
    ("Quebec", "1-866-277-3553 or 1-866-APPELLE is a toll-free, French-speaking helpline (also available in English)."):
        {
            "numbers_found": ["1-866-277-3553"],
            "fix_note": "'1-866-APPELLE' and '1-866-277-3553' are the SAME number - 'APPELLE' decodes to "
                        "'2773553' on a phone keypad, identical to the number already given in digits. "
                        "They are not two different numbers, so this is kept as a single entry rather than two.",
        },
}


def apply_manual_fixes(path_in, path_out):
    with open(path_in, encoding="utf-8") as f:
        data = json.load(f)

    applied = 0
    for country, entries in data["countries"].items():
        for entry in entries:
            key = (country, entry["text"])
            if key in MANUAL_FIXES:
                fix = MANUAL_FIXES[key]
                entry["numbers_found"] = fix["numbers_found"]
                entry["manual_fix_note"] = fix["fix_note"]
                applied += 1

    data["_meta"]["manual_fixes_applied"] = applied
    data["_meta"]["manual_fixes_note"] = (
        "Some entries (numbers with phone-keypad letters, or text ambiguous between "
        "schedules and numbers) were manually corrected after human review. "
        "See 'manual_fix_note' on each affected entry."
    )

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Manual fixes applied: {applied} of {len(MANUAL_FIXES)} expected")


if __name__ == "__main__":
    apply_manual_fixes("crisis_lines_full.json", "crisis_lines_full.json")
