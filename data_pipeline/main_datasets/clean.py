"""Content cleaning for converted datasets (data/processed -> data/clean).

Steps, in order:
  1. Assistant turns, sentence level:
     a. crisis resources (hotlines, crisis numbers/sites) -> {{CRISIS_LINE}},
        emergency numbers (911) -> {{EMERGENCY_NUMBER}}
     b. other phone numbers, URLs and emails -> sentence removed
        (the model must never learn to produce contacts or links)
     c. self-promotion, credentials and signatures -> sentence removed (PROMO_SOURCES);
        name signatures at the end of a sentence ("~Mark", "Mirella~") are cut off
     d. Psych8k artefacts of the GPT-4 rewrite: the formulaic opener
        "As a counsellor, ..." is stripped (rest of the sentence kept) and
        narration about the session ("I replied, '...'") is removed
     e. therapist names ("my name is Alan", "call me John"): the model is an
        AI and must not introduce itself with a human name
     f. ESConv (collected on MTurk): instructions about the task itself
        ("press quit and take the survey") removed
     g. laughter and emoticons ("lol", "haha", ":)", ":D") removed
  2. User turns: emails and phone numbers masked as [EMAIL] / [PHONE]
  3. Transcripts (AnnoMI, HOPE), both roles: spoken-language artefacts removed:
     non-verbal and transcription marks ([chuckles], [unintelligible 00:08:07]),
     fillers ("um", "uh"), stutters ("I-I-I", "that-that", "wh-what") and
     restarted clauses ("It really-- You're in the driver's seat")
  4. AnnoMI turn structure, as in a written chat:
     - therapist turns that are only a backchannel ("Mm-hmm.", "Yeah.") are
       removed and the client turns around them merged (kept when they
       answer a client question)
     - a therapist sentence cut by a client "Yeah." and continued in the next
       turn is joined back ("the effects of-" / "Yeah." / "-last night")
     - a sentence abandoned at the end of a turn is cut, keeping the complete
       sentences before it; a turn that is only a fragment is left as it is
       (it is masked in manual_edits.csv, step 8)
  5. Drop examples left with an empty assistant turn or a too-short
     single-turn answer
  6. Drop examples whose overall language is not English
  7. Exact deduplication across all datasets (normalized text)
     Steps 5-7 are decided on the text BEFORE step 8, so a manual edit never
     changes which conversations exist.
  8. Manual line-by-line review, data/review/manual_edits.csv: the assistant
     turn, matched by its full text after steps 1-4, is edited ("editar") or
     masked ("mascarar": "train": False, kept as context, excluded from the
     loss). Every mask comes from this file; no assistant turn is masked
     automatically. Rows that match nothing are listed in
     data/review/manual_edits_not_applied.jsonl

Every sentence-level change is logged to data/review/changes.jsonl.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict

from langdetect import DetectorFactory, LangDetectException, detect_langs

from common import PROCESSED_DIR, REPO_ROOT

DetectorFactory.seed = 0  # langdetect is non-deterministic without a fixed seed

CLEAN_DIR = REPO_ROOT / "data" / "clean"
REVIEW_DIR = REPO_ROOT / "data" / "review"
# Manual review decisions (self-disclosure, religion, diagnosis); see load_manual_edits()
MANUAL_EDITS_PATH = REVIEW_DIR / "manual_edits.csv"

# Priority order: when duplicates exist, the copy from the earlier source is kept
SOURCES = ["annomi", "esconv", "amod", "psych8k", "empathetic"]
EVAL_SOURCES = ["hope_eval"]  # cleaned but excluded from cross-dataset dedup

PROMO_SOURCES = {"amod"}
ROLE_PREFIX_SOURCES = {"psych8k"}
TRANSCRIPT_SOURCES = {"annomi", "hope"}
BACKCHANNEL_SOURCES = {"annomi"}
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
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|gov|edu|info|co\.uk)\b(?:/\S*)?"
    r"|(?<![\w.])www\.(?=\s|$)",  # URL broken by a space: "Check out www. Blog..."
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
    r"|\bmy\s+(?:most\s+)?(?:latest|recent|new)\s+(?:blog|post|article|book|video|newsletter)\b"
    r"|\bcheck\s+out\s+my\b|\bvia\s+skype\b|\bIn\s+it\s+I\s+offer\b"
    r"|\bI\s+can\s+offer\s+you\s+(?:hypnosis|therapy|counseling|counselling|sessions?)\b"
    r"|\bmy\s+book,?\s+living\b|\b(?:at|of)\s+my\s+book\b"  # not the idiom "in my book"
    r"|\bI(?:\s+am|'m)\s+an?\s+(?:licensed|certified|registered)\b"
    r"|\bfree\s+(?:consultation|session)\b"
    r"|\bcurrently\s+providing\s+services\b"
    r"|\b(?:LPC|LMFT|LCSW|LMSW|LMHC|LCPC|LCADC|LMAC|NCC|DBH|MFT|MSW|LPCC)\b",
    re.I,
)

# Signatures at the end of a sentence: "You are deserving!~Mark", "Best of luck. Mirella~"
SIGNATURE = re.compile(
    r"\s*(?:~\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\b[A-Z][a-z]+~|\bSincerely~?\s+[A-Z][a-z]+(?:\s+[A-Z]\w*)*)\s*$"
)

# Psych8k: formulaic opener added by the GPT-4 rewrite ("As a counsellor, I understand...")
ROLE_PREFIX = re.compile(r"^As\s+(?:a|the|your)\s+(?:counsell?or|therapist|psychologist),\s*", re.I)
# Psych8k: the answer narrates the session instead of being the answer
NARRATION = re.compile(r"\bI\s+(?:replied|responded|asked|said),\s*['\"‘“]", re.I)

# Therapist introducing themselves with a human name
NAME_CLAUSE = re.compile(  # removed from the sentence: "Hello, my name is Amber, how are you?"
    r"(?:,\s*|\s+)?\b[Mm]y\s+name\s+is\s+[A-Z][a-z]+\b"
    r"(?:,?\s+and\s+I\s+am\s+one\s+of\s+the\s+counsel+ors\s+here)?"
)
NAME_SENTENCE = re.compile(  # whole sentence removed
    r"\bcall\s+me\s+[A-Z][a-z]+\b|\b[Mm]y\s+name\s+is\s+[A-Z][a-z]+\s+and\s+I\s+work\b"
)

# Transcripts: non-verbal and transcription marks, incl. timestamps ("[unintelligible 00:08:07]")
NONVERBAL = re.compile(
    r"\[(?:(?:unintelligible|inaudible)(?:\s+\d{1,2}:\d{1,2}(?::\d{1,2})?)?"
    r"|chuckl\w*|laugh\w*|cross\s?talk|croostalk|sighs?|signs|cough\w*|clears?\s+throat"
    r"|silence|scoffs?|nods?|pause)\]",
    re.I,
)

# AnnoMI: therapist turn made only of backchannels ("Mm-hmm.", "Yeah. Okay.")
BACKCHANNEL = re.compile(
    r"^(?:(?:mm+-?hmm+|mhm+|uh-?huh|mm+|hmm+|um+|uh+|yeah|yes|yep|okay|ok|right|sure"
    r"|alright|all\s+right|oh|ah|i\s+see)[\s.,!?…-]*)+$",
    re.I,
)

# Transcripts: spoken disfluencies
FILLER = re.compile(  # not "uh-huh"; in "and, uh--" the "--" stays for RESTART / CUT_END
    r"(?:,\s*)?(?<![\w-])(?:u+m+|u+h+|e+r+m+)—?(?!\w|-\w)(?:\s*,)?", re.I)
FILLER_BEFORE_DASH = re.compile(r"(?:,\s*|\s+)(?:u+m+|u+h+|e+r+m+)(?=-{1,2}(?:\s|$))", re.I)  # "and, um--" -> "and--"
FILLER_START = re.compile(r"(^|[.!?]\s+|\n)(?:u+m+|u+h+|e+r+m+)(?:-{1,2}|—)?(?![\w-])[,.]?\s+(\w)", re.I)
STUTTER_PARTIAL =re.compile(r"\b(\w{1,3})-(?=\1\w)", re.I)               # "wh-what" -> "what"
STUTTER_WORDS = re.compile(r"\b((?:\w+'?\w*\s+){0,3}\w+'?\w*)(?:-{1,2}\s*\1\b)+", re.I)  # "I-I-I", "in this- in this"
REPEATED_WORD = re.compile(r"\b(\w+)(?:,\s*|\s+)\1\b", re.I)          # "I, I did" / "the the"
# Restarted clause: "It really-- You're in the driver's seat" -> "You're in the driver's seat"
RESTART = re.compile(r"(^|[.!?,;]\s+|\n)[^.!?,;\n\"]*?\w--\s+(\w)")
CUT_END = re.compile(r"\w-{1,2}\s*$")                                    # "...sitting here with me today-"
SENTENCE_END = re.compile(r"[.!?][\"')\]]?(?=\s)")

# ESConv (crowdsourced on MTurk): instructions about the task itself -> sentence removed
TASK_META = re.compile(
    r"\bsurvey\b|\b(?:end|quit|close|finish)\s+(?:the|this|our)\s+(?:chat|conversation|session)\b"
    r"|\bpress(?:ed|ing)?\s+(?:the\s+)?(?:quit|end|finish)\b|\bquit\s+button\b|\bclick\s+(?:on\s+)?quit\b"
    r"|\b\d+\s+messages\b|\bthe\s+hit\b|\bquit\s+(?:your|on\s+your|from\s+your)\b"
    r"|\b(?:need|have)\s+to\s+(?:be\s+the\s+one\s+to\s+)?(?:press\s+)?quit\b|\banswer\s+the\s+questions\b",
    re.I,
)
# Chat-style laughter and emoticons in assistant turns
EMOTICON = re.compile(
    r"(?<!\w)(?:lol+|lmao|rofl|ha(?:ha)+h?|he(?:he)+|xd)(?!\w)"
    r"|(?<![\w:;=])[:;=]-?[()DPpOo3/\\|]+(?![\w])|(?<!\w)<3(?!\w)|\(\s*:\s*\)"
    r"|[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]",
    re.I,
)

EMOTICON_BOUNDARY = re.compile(rf"(?<=\w)\s*(?:{EMOTICON.pattern})\s+(?=[A-Z])", re.I)

# Psych8k: "as a counsellor" in the middle of a sentence ("but as a counsellor, I won't...")
ROLE_CLAUSE = re.compile(r"\bas\s+(?:a|the|your)\s+counsell?or,\s*(?=(?:I|my)\b)", re.I)


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


def remove_name_clause(sentence):
    """'Hello, my name is Amber, how are you?' -> 'Hello, how are you?'"""
    s = NAME_CLAUSE.sub("", sentence)
    s = re.sub(r"^[\s,.;:!?-]+", "", s)
    s = re.sub(r"\s+([,.!?])", r"\1", s)
    s = re.sub(r",(?=[.!?]|$)", "", s)
    if not re.search(r"\w", s):
        return None
    if sentence[:1].isupper():  # "My name is Rob, how are you?" -> "How are you?"
        s = s[:1].upper() + s[1:]
    return s


def clean_sentence(sentence, source):
    """Returns (new_sentence_or_None, rules) where rules lists the rules that changed it."""
    rules = []
    s = EMERGENCY_NUM.sub(EMERGENCY, sentence)
    if CRISIS_CONTEXT.search(s):
        s = HOTLINE_NAME.sub(CRISIS_LINE, s)
        s = PHONE_WRAPPED.sub(CRISIS_LINE, s)
    else:
        s = PHONE_WRAPPED.sub(lambda m: CRISIS_LINE if is_known_crisis_number(m) else m.group(), s)
    s = collapse_placeholders(s)
    if s != sentence:
        rules.append("crisis_placeholder")

    # Broken URL glued after a finished sentence ('...help)."www.') is cut off, keeping the sentence
    unglued = re.sub(r"(?<=[.!?\"')])\s*\(?www\.\s*$", "", s)
    if unglued != s:
        s = unglued
        rules.append("broken_url_removed")

    # Anything still carrying a contact (incl. crisis websites) is removed
    if PHONE.search(s) or URL.search(s) or EMAIL.search(s):
        return None, rules + ["contact_removed"]
    if source in PROMO_SOURCES and PROMO.search(s):
        return None, rules + ["promo_removed"]
    if source in PROMO_SOURCES and SIGNATURE.search(s):
        s = re.sub(r",\s*$", ".", SIGNATURE.sub("", s))
        rules.append("signature_removed")
        if not re.search(r"\w", s):
            return None, rules
    if s.endswith("~"):  # stray tilde: "I wish you all the best~"
        s = s.rstrip("~").rstrip()
        rules.append("tilde_removed")

    if source == "esconv" and TASK_META.search(s):
        # "you're welcome (you need to press quit, I can't)" -> keep "you're welcome"
        no_paren = re.sub(r"\s*\([^()]*\)", lambda m: "" if TASK_META.search(m.group()) else m.group(), s).strip()
        if no_paren and not TASK_META.search(no_paren) and re.search(r"\w", no_paren):
            s = no_paren
            rules.append("task_meta_removed")
        else:
            return None, rules + ["task_meta_removed"]

    if source in ROLE_PREFIX_SOURCES:
        if NARRATION.search(s):
            return None, rules + ["narration_removed"]
        stripped = ROLE_CLAUSE.sub("", ROLE_PREFIX.sub("", s))
        if stripped != s:
            s = stripped[:1].upper() + stripped[1:]
            rules.append("role_prefix_removed")

    if NAME_SENTENCE.search(s):
        return None, rules + ["therapist_name_removed"]
    if NAME_CLAUSE.search(s):
        s = remove_name_clause(s)
        rules.append("therapist_name_removed")
    return s, rules


def tidy_spacing(text):
    """Fixes spacing and punctuation left behind after removing words."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([,.!?;:])", r"\1", text)
    text = re.sub(r",\s*([,.!?])", r"\1", text)        # ", ," / ", ."
    text = re.sub(r"([.!?])\s*,", r"\1", text)          # ". ,"
    text = re.sub(r"(^|\n)[ \t]*[,;:]\s*", r"\1", text)  # line starting with a comma
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_restarts(text):
    """Drops a clause abandoned with "--" and restarted: "Since the-- since you used" -> "Since you used"."""
    def fix(m):
        before, first = m.group(1), m.group(2)
        mid_sentence = before.strip() in (",", ";")
        if mid_sentence and first.isupper():  # restart is a new sentence: "now, but it's-- Like I said"
            return ". " + first
        if mid_sentence:                       # "I mean, that's-- that relieves" -> "I mean, that relieves"
            return before + first
        return before + first.upper()          # start of a sentence
    return RESTART.sub(fix, text)


def remove_fillers(text):
    """'Okay. Um, so I know' -> 'Okay. So I know'; 'what, um, you' -> 'what you'"""
    text = FILLER_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    text = FILLER_BEFORE_DASH.sub("", text)
    return FILLER.sub(" ", text)


def remove_nonverbal(text, source, rec_id, log, counts):
    """Transcripts only: drops [chuckles], [crosstalk], [unintelligible 00:08:07], ...,
    spoken fillers ("um", "uh") and stutters ("I-I-I", "that-that", "wh-what")."""
    if source not in TRANSCRIPT_SOURCES:
        return text
    new = text
    for rule, pattern, repl in (("nonverbal_removed", NONVERBAL, " "),
                                ("stutter_removed", STUTTER_PARTIAL, ""),
                                ("stutter_removed", STUTTER_WORDS, r"\1"),
                                ("filler_removed", FILLER, None),
                                ("stutter_removed", STUTTER_WORDS, r"\1"),  # again: "the, uh- the" -> "the- the"
                                ("stutter_removed", REPEATED_WORD, r"\1"),
                                ("restart_removed", RESTART, None)):
        if pattern.search(new):
            if rule == "filler_removed":
                new = remove_fillers(new)
            elif rule == "restart_removed":
                new = remove_restarts(new)
            else:
                new = pattern.sub(repl, new)
            counts[(source, rule)] += 1
    if new == text:
        return text
    new = tidy_spacing(new)
    log.append({"id": rec_id, "rule": "transcript_cleanup", "before": text, "after": new})
    return new


def remove_emoticons(text, source, rec_id, log, counts):
    """'Lol. Good luck anyway :)' -> 'Good luck anyway'"""
    if not EMOTICON.search(text):
        return text
    new = EMOTICON_BOUNDARY.sub(". ", text)  # "appearance :D I used to" -> "appearance. I used to"
    new = tidy_spacing(re.sub(r"(?<=[.!?])\s*[.!?]+", "", EMOTICON.sub(" ", new)))
    new = re.sub(r"\(\s*\)", "", new).strip()
    new = re.sub(r"(^|\n)[\s.,!?]+(?=\w)", r"\1", new)  # "hahaha...Have" -> "Have"
    if new and EMOTICON.match(text.lstrip()):  # "LOL if only..." -> "If only..."
        new = new[:1].upper() + new[1:]
    counts[(source, "emoticon_removed")] += 1
    log.append({"id": rec_id, "rule": "emoticon_removed", "before": text, "after": new})
    return new


def clean_assistant(text, source, rec_id, log, counts):
    """Returns the cleaned text; '' when nothing usable is left."""
    text = remove_nonverbal(text, source, rec_id, log, counts)
    text = remove_emoticons(text, source, rec_id, log, counts)
    out_lines = []
    for line in text.split("\n"):
        kept = []
        for sent in (p for p in SENT_SPLIT.split(line) if p.strip()):
            new, rules = clean_sentence(sent, source)
            if rules:
                for rule in rules:
                    counts[(source, rule)] += 1
                log.append({"id": rec_id, "rule": "+".join(rules), "before": sent, "after": new})
            if new:
                kept.append(new)
        if kept:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines).strip()


def tidy_transcript_turns(messages, source, rec_id, log, counts):
    """Transcripts: removes turns left empty by remove_nonverbal and, in AnnoMI,
    therapist turns that are only "Mm-hmm." / "Yeah." / "Okay."; the turns around
    them are merged, as in a written chat. A backchannel that answers a client
    question ("Is that normal?" -> "Yeah.") is kept."""
    if source not in TRANSCRIPT_SOURCES:
        return messages
    annomi = source in BACKCHANNEL_SOURCES
    out = []
    for i, m in enumerate(messages):
        if not m["content"]:
            counts[(source, "turn_empty_after_nonverbal")] += 1
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        # Therapist cut off by "Yeah." and continuing the same sentence in the next turn:
        # "the effects of-" / "Yeah." / "-last night" -> "the effects of last night"
        if (annomi and m["role"] == "user" and out and out[-1]["role"] == "assistant"
                and CUT_END.search(out[-1]["content"]) and BACKCHANNEL.match(m["content"].strip())
                and nxt is not None and nxt["role"] == "assistant" and nxt["content"]):
            counts[(source, "cut_sentence_joined")] += 1
            log.append({"id": rec_id, "rule": "cut_sentence_joined", "before": m["content"], "after": None})
            out[-1]["content"] = re.sub(r"-{1,2}\s*$", "", out[-1]["content"])
            out.append({"role": "assistant", "content": "", "_join": True})
            continue
        if (annomi and m["role"] == "assistant"
                and BACKCHANNEL.match(m["content"].strip())
                and out and not out[-1].get("_join")
                and not out[-1]["content"].rstrip().endswith("?")):
            counts[(source, "backchannel_removed")] += 1
            log.append({"id": rec_id, "rule": "backchannel_removed", "before": m["content"], "after": None})
            continue
        if out and out[-1].get("_join"):  # second half of a cut sentence
            out.pop()
            joined = f'{out[-1]["content"]} {re.sub(r"^-{1,2}", "", m["content"])}'
            out[-1]["content"] = tidy_spacing(remove_fillers(joined))
        elif out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = f'{out[-1]["content"]} {m["content"]}'
        else:
            out.append(dict(m))
    if annomi:
        for m in out:
            if m["role"] == "assistant":
                fix_cut_end(m, source, rec_id, log, counts)
    while out and out[0]["role"] != "user":  # chunk starting on a removed turn
        out.pop(0)
    while out and out[-1]["role"] != "assistant":  # chunk ending on a removed turn
        out.pop()
    return out


def fix_cut_end(m, source, rec_id, log, counts):
    """Therapist sentence abandoned at the end of the turn ("And, uh--"): the
    unfinished fragment is cut and the complete sentences before it are kept.
    A turn that is only a fragment is left unchanged (masked in manual_edits.csv)."""
    text = m["content"]
    if not CUT_END.search(text):
        return
    ends = list(SENTENCE_END.finditer(text))
    new = text[:ends[-1].end()].strip() if ends else ""
    # keep the complete part only if it is a real answer: quotes balanced, not just "Yeah."
    if new and new.count('"') % 2 == 0 and not BACKCHANNEL.match(new):
        counts[(source, "cut_fragment_removed")] += 1
        log.append({"id": rec_id, "rule": "cut_fragment_removed", "before": text, "after": new})
        m["content"] = new
    else:
        counts[(source, "cut_turn_kept")] += 1


def clean_user(text, source, rec_id, log, counts):
    text = remove_nonverbal(text, source, rec_id, log, counts)
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

def load_manual_edits(path=MANUAL_EDITS_PATH):
    """Reads the manually reviewed decisions, indexed by (conv_id, full assistant turn).
    Columns used: id, conv_id, turno_original (the whole assistant turn after the
    automatic cleaning), acao ("mascarar" | "editar"), procurar, substituir_por.
    - editar: "procurar" is replaced by "substituir_por" in that turn
    - mascarar: that turn gets "train": False (kept as context, never learned)
    - any other value is ignored
    The turn is matched by its full text, so a short "procurar" can never touch
    another turn of the same conversation."""
    if not path.exists():
        return {}
    edits = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            action = row["acao"].strip().lower()
            if action == "editar" and not row["procurar"]:
                continue
            if action in ("editar", "mascarar"):
                key = (row["conv_id"], row["turno_original"].strip())
                edits[key].append({**row, "acao": action})
    return edits


def apply_manual_edits(rec_id, source, conv_id, messages, edits, applied, log, counts):
    for m in messages:
        if m["role"] != "assistant":
            continue
        rows = edits.get((conv_id, m["content"].strip()))
        if not rows:
            continue
        original = m["content"]
        for e in rows:
            applied.add(e["id"])
            if e["acao"] == "mascarar":
                m["train"] = False
                counts[(source, "manual_mask")] += 1
            elif e["procurar"] in m["content"]:
                m["content"] = m["content"].replace(e["procurar"], e["substituir_por"])
                counts[(source, "manual_edit")] += 1
        if m["content"] != original:
            m["content"] = tidy_spacing(m["content"])
            log.append({"id": rec_id, "rule": "manual_edit", "before": original, "after": m["content"]})
            if not re.search(r"\w", m["content"]):  # an edit emptied the turn: undone, use "mascarar" instead
                m["content"] = original
                counts[(source, "manual_edit_emptied_turn_undone")] += 1


def clean_record(rec, log, counts, edits=None, applied=None):
    """Returns (cleaned_record, None) or (None, drop_reason).
    If an assistant turn ends up empty (it was only a link or contact), the
    conversation is truncated right before that exchange."""
    source = rec["source"]
    messages = []
    for m in rec["messages"]:
        if m["role"] == "assistant":
            content = clean_assistant(m["content"], source, rec["id"], log, counts)
            only_nonverbal = source in TRANSCRIPT_SOURCES and not NONVERBAL.sub("", m["content"]).strip()
            only_emoticon = not re.sub(r"[\s.,!?]", "", EMOTICON.sub("", m["content"]))
            if not content and only_emoticon:
                # a turn that is only ":)" or "lol": kept as it is (masked in manual_edits.csv)
                content = m["content"].strip()
                counts[(source, "emoticon_only_turn_kept")] += 1
            elif not content and not only_nonverbal:  # "[chuckles]" alone is tidied below instead
                messages.pop()  # drop the user turn left without an answer
                counts[(source, "truncated_at_empty_assistant")] += 1
                break
        else:
            content = clean_user(m["content"], source, rec["id"], log, counts)
        messages.append({"role": m["role"], "content": content, })

    messages = tidy_transcript_turns(messages, source, rec["id"], log, counts)
    # Filters and flags are decided on the automatically cleaned text, BEFORE the
    # manual edits: mix.py selects (and samples) conversations from them, so a
    # manual edit must never change which conversations exist or are selected.
    if len(messages) < 2:
        return None, "empty_assistant_turn"
    if (source in SINGLE_TURN_SOURCES
            and len(messages[-1]["content"].split()) < MIN_WORDS_SINGLE_TURN):
        return None, "too_short_answer"

    full_text = " ".join(m["content"] for m in messages)[:2000]
    if not is_english(full_text):
        return None, "not_english"

    meta = rec["meta"]
    for m in messages:
        if m["role"] == "assistant":
            m.setdefault("train", True)
    key = dedup_key({"messages": messages})  # before the manual edits, see step 7

    if edits:
        apply_manual_edits(rec["id"], source, rec["conv_id"], messages, edits,
                           set() if applied is None else applied, log, counts)
    return {**rec, "messages": messages, "meta": meta, "_dedup_key": key}, None


def main():
    log, counts = [], Counter()
    edits, applied = load_manual_edits(), set()
    stats = defaultdict(dict)
    seen, cleaned_all = set(), {}

    for name in SOURCES + EVAL_SOURCES:
        records = load(name)
        stats[name]["input"] = len(records)
        kept, drops = [], Counter()
        for rec in records:
            new, reason = clean_record(rec, log, counts, edits, applied)
            if new is None:
                drops[reason] += 1
                continue
            key = new.pop("_dedup_key")
            if name in SOURCES:
                if key in seen:
                    drops["duplicate"] += 1
                    continue
                seen.add(key)
            kept.append(new)
        stats[name].update({f"dropped: {k}": v for k, v in drops.items()})
        stats[name]["assistant turns excluded from training (train=False)"] = sum(
            m.get("train") is False for r in kept for m in r["messages"])
        stats[name]["output"] = len(kept)
        cleaned_all[name] = kept

    for name, records in cleaned_all.items():
        write(records, CLEAN_DIR / f"{name}.jsonl")
    write(log, REVIEW_DIR / "changes.jsonl")

    for (source, rule), n in sorted(counts.items()):
        key = source if source in stats else f"{source}_eval"
        stats[key][f"sentences: {rule}"] = n
    # Manual decisions that matched nothing (text changed upstream?): listed for checking
    not_applied = [e for rows in edits.values() for e in rows if e["id"] not in applied]
    not_applied = list({e["id"]: e for e in not_applied}.values())
    write(not_applied, REVIEW_DIR / "manual_edits_not_applied.jsonl")
    stats["manual_edits"] = {"rows": sum(len(r) for r in edits.values()), "not applied": len(not_applied)}

    with open(REVIEW_DIR / "clean_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    for name in SOURCES + EVAL_SOURCES:
        print(f"\n[{name}]")
        for k, v in stats[name].items():
            print(f"  {k}: {v}")
    print(f"\n[manual_edits] {stats['manual_edits']}")
    print(f"\nLogs: {REVIEW_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
