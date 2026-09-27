"""Content cleaning for converted datasets (data/processed -> data/clean).

Steps, in order:
  1. Assistant turns, sentence level:
     a. crisis resources (hotlines, crisis numbers/sites) -> {{CRISIS_LINE}},
        emergency numbers (911) -> {{EMERGENCY_NUMBER}}
     b. other phone numbers, URLs and emails -> sentence removed
        (the model must never learn to produce contacts or links)
     c. self-promotion, credentials and signatures -> sentence removed (PROMO_SOURCES)
  2. User turns: emails and phone numbers masked as [EMAIL] / [PHONE]
  3. Drop examples left with an empty assistant turn or a too-short
     single-turn answer
  4. Drop examples whose overall language is not English
  5. Exact deduplication across all datasets (normalized text)
  6. Flag examples whose user turns contain crisis language for manual review
     (kept, with meta["crisis_flag"] = True; mix.py decides what to do)
  7. Self-disclosure (the listener talking about their own life, which a
     virtual therapist would be inventing):
     - ESConv: assistant turns annotated "Self-disclosure" get "train": False
       (kept as context, excluded from the loss at training time)
     - EmpatheticDialogues: no annotation, so a heuristic sets
       meta["self_disclosure"] = True; mix.py prefers conversations without it

Every sentence-level change is logged to data/review/changes.jsonl.
"""
import hashlib
import json
import re
from collections import Counter, defaultdict

from langdetect import DetectorFactory, LangDetectException, detect_langs

from common import PROCESSED_DIR, REPO_ROOT

DetectorFactory.seed = 0  # langdetect is non-deterministic without a fixed seed

CLEAN_DIR = REPO_ROOT / "data" / "clean"
REVIEW_DIR = REPO_ROOT / "data" / "review"

# Priority order: when duplicates exist, the copy from the earlier source is kept
SOURCES = ["annomi", "esconv", "amod", "psych8k", "empathetic"]
EVAL_SOURCES = ["hope_eval"]  # cleaned but excluded from cross-dataset dedup

PROMO_SOURCES = {"amod"}
SINGLE_TURN_SOURCES = {"amod", "psych8k"}
MIN_WORDS_SINGLE_TURN = 15

CRISIS_LINE = "{{CRISIS_LINE}}"
EMERGENCY = "{{EMERGENCY_NUMBER}}"

# --- Patterns -----------------------------------------------------------------

_PHONE_CORE = (
    r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s?|\b\d{3}[-.\s])\d{3}[-.\s]\d{4}\b"  # 555-555-5555, (555) 555-5555
    r"|\b1-?8\d\d-?[A-Z0-9]{3}-?[A-Z0-9]{4}\b"                              # 1-800-XXX-XXXX
    r"|\b1-8\d\d(?:-[A-Z0-9]{1,5}){2,}\b"                                     # vanity: 1-800-4-A-CHILD
    r"|\b741[- ]?741\b|\b988\b"                                             # crisis text line, US 988
)
PHONE = re.compile(_PHONE_CORE)
PHONE_WRAPPED = re.compile(rf"\(\s*(?:{_PHONE_CORE})\s*\)|{_PHONE_CORE}")
# Crisis numbers seen in the data; replaced even when the sentence itself has no crisis keyword
KNOWN_CRISIS_NUMBERS = {
    "8002738255",  # US National Suicide Prevention Lifeline (old number)
    "8004224453", "8004achild",  # Childhelp
    "8775658860",  # Trans Lifeline
    "8664887386",  # Trevor Project
    "8887247240",  # domestic violence hotline
    "8009896884",  # Texas crisis line
    "741741", "988",
}
EMERGENCY_NUM = re.compile(r"\b911\b")
URL = re.compile(
    r"https?://\S+|(?<![\w.])www\.\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|gov|edu|info|co\.uk)\b(?:/\S*)?",
    re.I,
)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

CRISIS_CONTEXT = re.compile(
    r"suicid|crisis|hotline|lifeline|helpline|help line|text line|prevention"
    r"|trevor project|samaritans|741741|\b988\b"
    r"|want(?:s|ed)?\s+to\s+die|kill(?:ing)?\s+(?:your|my|him|her)sel(?:f|ves)"
    r"|harm(?:ing)?\s+(?:your|my)sel(?:f|ves)|immediate\s+(?:help|danger|support)|emergency",
    re.I,
)
# Only named services are replaced; generic advice ("call a crisis hotline") is kept
HOTLINE_NAME = re.compile(
    r"\b(?:the\s+)?national\s+(?:suicide(?:\s+prevention)?\s+|crisis\s+)?(?:life|hot|help)line\b"
    r"|\b(?:the\s+)?(?:trevor project|trans lifeline|crisis text line)\b"
    r"|(?-i:\b(?:[Tt]he\s+)?(?:[A-Z][a-z]+\s+)+(?:Crisis|Suicide)\s+(?:Hot|Help|Life)line\b)",
    re.I,
)

PROMO = re.compile(
    r"\b(?:feel free(?: to)?|please|hesitate to|you can|you may|welcome to|and)\s+"
    r"(?:\w+\s+)?(?:contact|reach|call|email|e-mail|message)\s+me\b"
    r"|\bmy\s+(?:practice|website|site|blog|podcast|clinic|youtube)\b"
    r"|\bmy\s+book,?\s+living\b|\b(?:at|of)\s+my\s+book\b"  # not the idiom "in my book"
    r"|\bI(?:\s+am|'m)\s+an?\s+(?:licensed|certified|registered)\b"
    r"|\bfree\s+(?:consultation|session)\b"
    r"|\bcurrently\s+providing\s+services\b"
    r"|\b(?:LPC|LMFT|LCSW|LMSW|LMHC|LCPC|LCADC|LMAC|NCC|DBH|MFT|MSW|LPCC)\b",
    re.I,
)

# Crisis language in user turns (flagging only, not removal)
CRISIS_USER = re.compile(
    r"\bsuicid\w*|\bkill(?:ing)?\s+my\s*self\b|\bend(?:ing)?\s+(?:my|it)\s+(?:life|all)\b"
    r"|\bwant(?:ed)?\s+to\s+die\b|\bbetter\s+off\s+dead\b|\bno\s+reason\s+to\s+live\b"
    r"|\bself[-\s]?harm\w*|\bcut(?:ting)?\s+my\s*self\b|\bhurt(?:ing)?\s+my\s*self\b"
    r"|\boverdos\w*|\btake\s+my\s+(?:own\s+)?life\b|\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)\b",
    re.I,
)

# Heuristic for EmpatheticDialogues (no annotation available); errs on the side of flagging
SELF_DISCLOSURE = re.compile(
    r"\b(?:I|I've|I'm|I had|I was)\b[^.?!]*\b(?:too|also|as well|same thing|myself)\b"
    r"|\bmy\s+(?:wife|husband|son|daughter|kids?|mom|dad|mother|father|brother|sister"
    r"|dog|cat|friend|boyfriend|girlfriend|job|boss)\b"
    r"|\bwhen I was\b|\bI\s+(?:have|had|got|went|used to|remember)\b",
    re.I,
)

# Sentence split: after .!? followed by whitespace, or glued sentences like "you.I hope"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[a-z0-9][.!?])(?=[A-Z][a-z])")


# --- Helpers ------------------------------------------------------------------

def is_english(text, min_prob=0.9):
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return True  # no letters to judge -> keep
    top = langs[0]
    return top.lang == "en" or top.prob < min_prob


def collapse_placeholders(text):
    for ph in (CRISIS_LINE, EMERGENCY):
        esc = re.escape(ph)
        text = re.sub(rf"{esc}\s*\(\s*{esc}\s*\)", ph, text)
        # "{{X}}: {{X}}", "{{X}} ({{X}})", "{{X}} or {{X}}" ... -> "{{X}}"
        text = re.sub(rf"{esc}(?:(?:\W{{0,4}}|\s+(?:at|on|or|is)\s+){esc})+", ph, text)
    return text


def is_known_crisis_number(match):
    key = re.sub(r"[^0-9a-z]", "", match.group().lower())
    if len(key) == 11 and key.startswith("1"):
        key = key[1:]
    return key in KNOWN_CRISIS_NUMBERS


def clean_sentence(sentence, source):
    """Returns (new_sentence_or_None, rule) where rule is None if unchanged."""
    s = EMERGENCY_NUM.sub(EMERGENCY, sentence)
    if CRISIS_CONTEXT.search(s):
        s = HOTLINE_NAME.sub(CRISIS_LINE, s)
        s = PHONE_WRAPPED.sub(CRISIS_LINE, s)
    else:
        s = PHONE_WRAPPED.sub(lambda m: CRISIS_LINE if is_known_crisis_number(m) else m.group(), s)
    s = collapse_placeholders(s)

    # Anything still carrying a contact (incl. crisis websites) is removed
    if PHONE.search(s) or URL.search(s) or EMAIL.search(s):
        return None, "contact_removed"
    if source in PROMO_SOURCES and PROMO.search(s):
        return None, "promo_removed"
    return s, ("crisis_placeholder" if s != sentence else None)


def clean_assistant(text, source, rec_id, log, counts):
    out_lines = []
    for line in text.split("\n"):
        kept = []
        for sent in (p for p in SENT_SPLIT.split(line) if p.strip()):
            new, rule = clean_sentence(sent, source)
            if rule:
                counts[(source, rule)] += 1
                log.append({"id": rec_id, "rule": rule, "before": sent, "after": new})
            if new:
                kept.append(new)
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines).strip()


def clean_user(text, source, rec_id, log, counts):
    new = EMAIL.sub("[EMAIL]", text)
    new = PHONE.sub("[PHONE]", new)
    if new != text:
        counts[(source, "user_pii_masked")] += 1
        log.append({"id": rec_id, "rule": "user_pii_masked", "before": text, "after": new})
    return new


def dedup_key(record):
    text = " ".join(m["content"] for m in record["messages"]).lower()
    return hashlib.md5(re.sub(r"[^a-z0-9]", "", text).encode()).hexdigest()


def load(name):
    with open(PROCESSED_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- Pipeline -----------------------------------------------------------------

def clean_record(rec, log, counts):
    """Returns (cleaned_record, None) or (None, drop_reason).
    If an assistant turn ends up empty (it was only a link or contact), the
    conversation is truncated right before that exchange."""
    source = rec["source"]
    messages = []
    for m in rec["messages"]:
        if m["role"] == "assistant":
            content = clean_assistant(m["content"], source, rec["id"], log, counts)
            if not content:
                messages.pop()  # drop the user turn left without an answer
                counts[(source, "truncated_at_empty_assistant")] += 1
                break
        else:
            content = clean_user(m["content"], source, rec["id"], log, counts)
        messages.append({"role": m["role"], "content": content})

    if len(messages) < 2:
        return None, "empty_assistant_turn"
    if (source in SINGLE_TURN_SOURCES
            and len(messages[-1]["content"].split()) < MIN_WORDS_SINGLE_TURN):
        return None, "too_short_answer"

    full_text = " ".join(m["content"] for m in messages)[:2000]
    if not is_english(full_text):
        return None, "not_english"

    flagged = any(m["role"] == "user" and CRISIS_USER.search(m["content"]) for m in messages)
    meta = {**rec["meta"], "crisis_flag": flagged}
    mark_self_disclosure(source, messages, meta)
    return {**rec, "messages": messages, "meta": meta}, None


def mark_self_disclosure(source, messages, meta):
    """Sets "train": False on assistant turns that should not be learned from."""
    assistant = [m for m in messages if m["role"] == "assistant"]
    if source == "esconv":
        # strategies are aligned with assistant turns (prefix-aligned after truncation)
        for m, strategies in zip(assistant, meta["strategies"]):
            if "Self-disclosure" in strategies:
                m["train"] = False
    elif source == "empathetic":
        meta["self_disclosure"] = any(SELF_DISCLOSURE.search(m["content"]) for m in assistant)
    for m in assistant:
        m.setdefault("train", True)


def main():
    log, counts = [], Counter()
    stats = defaultdict(dict)
    seen, cleaned_all, flagged = set(), {}, []

    for name in SOURCES + EVAL_SOURCES:
        records = load(name)
        stats[name]["input"] = len(records)
        kept, drops = [], Counter()
        for rec in records:
            new, reason = clean_record(rec, log, counts)
            if new is None:
                drops[reason] += 1
                continue
            if name in SOURCES:
                key = dedup_key(new)
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)
            kept.append(new)
            if new["meta"]["crisis_flag"]:
                flagged.append({"id": new["id"], "source": new["source"], "reviewed": None,
                                "decision": None, "messages": new["messages"]})
        stats[name].update({f"dropped: {k}": v for k, v in drops.items()})
        stats[name]["crisis_flagged"] = sum(r["meta"]["crisis_flag"] for r in kept)
        stats[name]["assistant turns excluded from training (train=False)"] = sum(
            m.get("train") is False for r in kept for m in r["messages"])
        if name == "empathetic":
            stats[name]["conversations with self-disclosure"] = sum(r["meta"]["self_disclosure"] for r in kept)
        stats[name]["output"] = len(kept)
        cleaned_all[name] = kept

    for name, records in cleaned_all.items():
        write(records, CLEAN_DIR / f"{name}.jsonl")
    write(log, REVIEW_DIR / "changes.jsonl")
    # Keep manual review decisions from previous runs (re-running clean.py must not erase them)
    flagged_path = REVIEW_DIR / "crisis_flagged.jsonl"
    if flagged_path.exists():
        with open(flagged_path, encoding="utf-8") as f:
            previous = {r["id"]: r for r in map(json.loads, f) if r.get("decision") is not None}
        for r in flagged:
            if r["id"] in previous:
                r["reviewed"] = previous[r["id"]].get("reviewed")
                r["decision"] = previous[r["id"]]["decision"]
    write(flagged, flagged_path)

    for (source, rule), n in sorted(counts.items()):
        key = source if source in stats else f"{source}_eval"
        stats[key][f"sentences: {rule}"] = n
    with open(REVIEW_DIR / "clean_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for name in SOURCES + EVAL_SOURCES:
        print(f"\n[{name}]")
        for k, v in stats[name].items():
            print(f"  {k}: {v}")
    print(f"\nLogs: {REVIEW_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
