#!/usr/bin/env python3
"""
Extracts the "List of suicide crisis lines" table from Wikipedia into JSON,
keeping ALL entries per country (not just one manually picked entry).

Usage:
  python3 extract_crisis_lines.py --live      # fetches the current Wikipedia page (needs internet)
  python3 extract_crisis_lines.py --from-file raw_table.txt   # uses an already-saved local snapshot

Dependencies for --live mode: pip install requests beautifulsoup4
"""
import re
import json
import argparse
from datetime import date

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_suicide_crisis_lines"

GENERIC_SERVICE_RE = re.compile(
    r"\b(police|fire brigades?|fire trucks?|fire stations?|ambulances?|gendarmerie|"
    r"emergency numbers?|emergency services?)\b",
    re.I,
)
MENTAL_HEALTH_RE = re.compile(
    r"\b(suicide|crisis|mental health|distress|depression|anxiety|emotional support|samaritan|lifeline|"
    r"helpline|hotline|befriend|counsel|therapy|psycholog|self[\s-]?harm|listening|hope|esperan[çc]a|"
    r"amiga|amigo|apoio|voz|amizade)\b",
    re.I,
)


def is_mental_health_relevant(text):
    """Keeps the entry if it mentions mental health/crisis, OR if it isn't
    clearly a generic emergency service (police/fire/ambulance) unrelated to it."""
    if MENTAL_HEALTH_RE.search(text):
        return True
    if GENERIC_SERVICE_RE.search(text):
        return False
    return True


PHONE_RE = re.compile(
    r"""
    (?:
        (?:\(?\+?\d{1,4}\)?[\s.\-–]?)?   # optional country code, with or without parentheses (e.g. "(+592)")
        \(?\d{2,5}\)?                    # start of the number / area code
        (?:[\s.\-–]?\d{2,7}(?!/|:\d)){1,4}   # following blocks (up to 7 digits each, never swallow a block
                                              # immediately followed by '/' (e.g. "24/7") or ':digit' (e.g. "11:00")
    )
    |
    \b\d{2,6}\b                       # isolated short numbers (e.g. 988, 911, 112, 1411, 43)
    """,
    re.VERBOSE,
)


def fetch_live_rows():
    """Fetches and parses the table directly from Wikipedia (needs internet)."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(WIKI_URL, headers={"User-Agent": "virtual-therapist-thesis-research/1.0"}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # The country table is the first wikitable on the page
    table = soup.find("table", class_="wikitable")
    rows = []
    for tr in table.find_all("tr")[1:]:  # skip the header
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        country = cells[0].get_text(strip=True)
        lines_cell = cells[1].get_text(separator=" ", strip=True)
        rows.append((country, lines_cell))
    return rows


def parse_from_file(path):
    """Parses a local snapshot saved in the '| Country | Lines |' format, one row per line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            country = parts[0]
            # join everything else back together (in case the text itself
            # contains a literal '|', as happened with "Cayman Islands" -
            # can't cut there)
            lines_cell = "|".join(parts[1:])
            rows.append((country, lines_cell))
    return rows


def split_entries(lines_cell):
    """Splits a country's raw text into individual entries (one per line/organisation),
    filtering out generic emergency services (police, fire, ambulance) unrelated to
    mental health/crisis."""
    raw_entries = [e.strip() for e in lines_cell.split(";") if e.strip()]
    entries = []
    for raw in raw_entries:
        if not is_mental_health_relevant(raw):
            continue

        numbers = []
        for m in PHONE_RE.finditer(raw):
            tok = m.group().strip()
            start, end = m.span()
            after = raw[end:end + 6]
            before = raw[max(0, start - 15):start]

            # skip times of day (e.g. "24 hours", "12 am", "7 days", "24/7", "24-hour")
            if re.match(r"[\s-]*(am|pm|hours?|hrs?|days?|:00)\b", after, re.I):
                continue
            if after.startswith("/"):
                continue
            if raw[max(0, start - 1):start] == ":":
                continue  # the minutes part of a time (e.g. "19:00"), not a phone number
            if re.match(r"^-", raw[start - 1:start] + "-") and start >= 1 and raw[start - 1] == "-" and start >= 2 and raw[start - 2].isalpha():
                continue  # suffix of a compound term like "COVID-19", not a phone number
            if re.search(r"\b(aged|under)\s*\d*[\s\-–]*$", before, re.I) or re.match(r"\s*[\s\-–]*years?\b", after, re.I):
                continue  # an age (e.g. "aged 5-25", "under 18"), not a phone number
            if re.search(r"\b(from|to|between|and)\s*$", before, re.I) and len(re.sub(r"\D", "", tok)) <= 2:
                continue
            if len(re.sub(r"\D", "", tok)) < 2:
                continue
            numbers.append(tok)

        seen = set()
        deduped = []
        for n in numbers:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        entries.append({"text": raw, "numbers_found": deduped})
    return entries


def build_dataset(rows):
    countries = {}
    for country, lines_cell in rows:
        countries[country] = split_entries(lines_cell)
    return {
        "_meta": {
            "description": "All lines/numbers per country, automatically extracted from Wikipedia. "
                            "Each country has a LIST of entries (not just one) - it's up to whoever "
                            "integrates this to pick which one to use, or show more than one.",
            "source": WIKI_URL,
            "extraction_date": str(date.today()),
            "warning": "Automatic extraction, not manually verified entry by entry. "
                       "Confirm critical numbers before production use.",
        },
        "countries": countries,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="Fetch the current Wikipedia page")
    ap.add_argument("--from-file", type=str, help="Use a local snapshot instead of hitting the internet")
    ap.add_argument("--out", type=str, default="crisis_lines_full.json")
    args = ap.parse_args()

    if args.live:
        rows = fetch_live_rows()
    elif args.from_file:
        rows = parse_from_file(args.from_file)
    else:
        raise SystemExit("Use --live or --from-file <path>")

    dataset = build_dataset(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    n_countries = len(dataset["countries"])
    n_entries = sum(len(v) for v in dataset["countries"].values())
    print(f"OK: {n_countries} countries, {n_entries} entries total -> {args.out}")
