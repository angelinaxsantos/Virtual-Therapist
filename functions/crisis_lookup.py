"""
functions/crisis_lookup.py

Deterministic resolution of contact placeholders in the model's responses.

The fine-tuned model never generates phone numbers. When it needs to point
the user to help, it writes one of two placeholders:
    {{CRISIS_LINE}}       -> a crisis / emotional support line
    {{EMERGENCY_NUMBER}}  -> the general emergency number (e.g. 112)

This module replaces them with the real contact for the user's country,
taken from data/crisis_lines_full.json (crisis lines) and EMERGENCY_NUMBERS
below (emergency numbers).

Typical usage inside the main conversation loop:

    from functions.crisis_lookup import resolve_crisis_placeholder

    response = fine_tuned_model.generate(...)
    response = resolve_crisis_placeholder(response, user_country, lang="pt", age=user_age)
"""
import json
from pathlib import Path

CRISIS_LINES_PATH = Path(__file__).resolve().parent.parent / "data" / "crisis_lines_full.json"
PLACEHOLDER = "{{CRISIS_LINE}}"
EMERGENCY_PLACEHOLDER = "{{EMERGENCY_NUMBER}}"

# Manual index: which of the several entries for each country (see
# crisis_lines_full.json) is the "primary" one shown by default. Fill this
# in as you confirm priority countries - never pick an entry with no numbers
# (see _pick_entry).
PRIMARY_ENTRY_INDEX = {
    "Portugal": 1,        # SOS Voz Amiga
    "United States": 1,   # 988 Suicide & Crisis Lifeline
    "United Kingdom": 4,  # Samaritans, 116 123
    "Brazil": 0,          # CVV
    "Spain": 0,           # Línea 024
}

# General emergency numbers (not crisis lines). 112 works across the whole EU.
_EU_112 = {
    "Portugal", "Spain", "France", "Germany", "Italy", "Greece", "Ireland",
    "Netherlands", "Belgium", "Austria", "Poland", "Sweden", "Denmark",
    "Finland", "Czech Republic", "Romania", "Hungary", "Luxembourg",
}
EMERGENCY_NUMBERS = {
    **{country: "112" for country in _EU_112},
    "United Kingdom": "999",
    "United States": "911",
    "Canada": "911",
    "Brazil": "192",
    "Australia": "000",
}

# Note: these values are actual user-facing content shown to the person
# (not code comments), so they intentionally stay in the target language.
FALLBACK_MESSAGE = {
    "pt": "uma linha de apoio verificada em befrienders.org ou findahelpline.com",
    "en": "a verified support line at befrienders.org or findahelpline.com",
}
EMERGENCY_FALLBACK = {
    "pt": "os serviços de emergência locais",
    "en": "your local emergency services",
}

_cache = None


def _load():
    global _cache
    if _cache is None:
        with open(CRISIS_LINES_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def _pick_entry(country: str, age: int | None = None) -> dict | None:
    """Picks the entry to use for a country.

    If `age` is given, first tries to find a line meant for that audience
    (e.g. elderly, 65+; children/youth, <18) using each entry's "audience"
    field. If there's no audience-specific line for that country (the most
    common case - see note below), falls back to the default behaviour:
    PRIMARY_ENTRY_INDEX, or the first entry that has a number.

    Note: as of this implementation, out of 214 lines across 102 countries,
    only 1 is specific to the elderly (Greece) - so for the vast majority of
    countries, this will always fall back to default behaviour even when an
    age is passed. This isn't a bug, it's a real limit of the publicly
    available data.
    """
    countries = _load()["countries"]
    entries = countries.get(country)
    if not entries:
        return None

    if age is not None:
        target_audience = None
        if age >= 65:
            target_audience = "elderly"
        elif age < 18:
            target_audience = "children_youth"

        if target_audience:
            for entry in entries:
                if target_audience in entry.get("audience", []) and entry["numbers_found"]:
                    return entry
            # no audience-specific line for this country -> fall back to default behaviour

    idx = PRIMARY_ENTRY_INDEX.get(country)
    if idx is not None and idx < len(entries) and entries[idx]["numbers_found"]:
        return entries[idx]

    for entry in entries:
        if entry["numbers_found"]:
            return entry

    return None  # no entry for this country has a usable number


def get_crisis_line_text(country: str, lang: str = "pt", age: int | None = None) -> str:
    """Returns the text ready to insert in place of {{CRISIS_LINE}}."""
    entry = _pick_entry(country, age=age)
    if entry is None:
        return FALLBACK_MESSAGE.get(lang, FALLBACK_MESSAGE["en"])
    return ", ".join(entry["numbers_found"])


def get_emergency_number_text(country: str, lang: str = "pt") -> str:
    """Returns the text ready to insert in place of {{EMERGENCY_NUMBER}}."""
    number = EMERGENCY_NUMBERS.get(country)
    return number if number else EMERGENCY_FALLBACK.get(lang, EMERGENCY_FALLBACK["en"])


def resolve_crisis_placeholder(response_text: str, country: str, lang: str = "pt", age: int | None = None) -> str:
    """Replaces the {{CRISIS_LINE}} and {{EMERGENCY_NUMBER}} placeholders in the
    model's response with real contacts. Text without placeholders is returned unchanged."""
    if PLACEHOLDER in response_text:
        response_text = response_text.replace(PLACEHOLDER, get_crisis_line_text(country, lang, age=age))
    if EMERGENCY_PLACEHOLDER in response_text:
        response_text = response_text.replace(EMERGENCY_PLACEHOLDER, get_emergency_number_text(country, lang))
    return response_text


if __name__ == "__main__":
    example = "Please reach out to {{CRISIS_LINE}} right now, they're trained to help."
    print(resolve_crisis_placeholder(example, "Portugal"))
    print(resolve_crisis_placeholder(example, "United States"))
    print(resolve_crisis_placeholder(example, "Ruritania", lang="en"))  # non-existent country

    print()
    print("--- With age (audience-based personalization) ---")
    print("Greece, age 70 (should pick the elderly line):")
    print(" ", get_crisis_line_text("Greece", age=70))
    print("Greece, age 30 (should pick the general line, not elderly):")
    print(" ", get_crisis_line_text("Greece", age=30))
    print("Portugal, age 70 (no elderly line available -> falls back to default):")
    print(" ", get_crisis_line_text("Portugal", age=70))

    print()
    print("--- Emergency number ---")
    emergency = "If you are in immediate danger, call {{EMERGENCY_NUMBER}} or {{CRISIS_LINE}}."
    print(resolve_crisis_placeholder(emergency, "Portugal"))
    print(resolve_crisis_placeholder(emergency, "United States", lang="en"))
    print(resolve_crisis_placeholder(emergency, "Ruritania", lang="en"))
