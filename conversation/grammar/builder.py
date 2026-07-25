"""
builder.py
===========
Rule-based Grammar & Sentence Builder (Stage 2).

PROBLEM THIS SOLVES: our SLR model recognizes isolated sign glosses in
the order they were signed. Sign language grammar differs from
English word order in well-documented ways -- WH-question words often
come at the END of a signed phrase rather than the start ("YOU NAME
WHAT" means "What is your name?"), and sign language typically omits
articles, copulas ("to be"), and auxiliary verbs that English
requires. Left as raw glosses, a recognized phrase like
["YOU", "NAME", "WHAT"] reads as gibberish to someone who doesn't sign.

WHY RULE-BASED, NOT A TRAINED MODEL: there's no parallel gloss-to-
English training data available for Eswatini Sign Language here, so a
trained seq2seq model isn't an option yet. This is a deliberately
narrow, hand-built set of patterns -- covers common, well-documented
cases well; will get plenty of other cases wrong. That's an honest
tradeoff, not a hidden one -- see LIMITATIONS below.

WHAT IT DOES:
  - Detects pronoun immediately followed by a "possessable" noun and
    converts the pronoun to possessive form ("YOU NAME" -> "your name").
  - Detects a WH-word anywhere in the phrase and, if present (or if
    the finalized turn was already tagged intent="question" by
    conversation/intent.py), reorders into a WH-initial question,
    inserting the appropriate copula for any pronoun subject found.
  - For non-questions, inserts a copula after a leading pronoun subject
    UNLESS the following word is a known common verb (see lexicon.py's
    COMMON_VERBS) -- "YOU HAPPY" -> "You are happy", but "YOU GO"
    stays "You go", not "You are go".
  - Handles a simple negation case: NOT/NO/NEVER anywhere in the
    phrase gets converted to "not" inserted after a copula/pronoun if
    one is present, or prefixed as "No, ..." otherwise.
  - Capitalizes the first letter and appends punctuation based on the
    tagged intent (? for questions, ! for greeting/farewell/thanks, .
    otherwise).

LIMITATIONS (explicitly NOT handled, don't assume otherwise):
  - Articles ("a"/"an"/"the") are not inserted at all. Reliably
    knowing which nouns need an article and which are definite vs.
    indefinite needs real NLP, not a hand list -- deferred.
  - Verb tense/aspect is not touched. Sign languages often use
    separate time-marker signs (FINISH, WILL, etc.) rather than verb
    conjugation; this builder doesn't yet look for or use those.
  - Plurals are not inserted.
  - Any verb gloss not in the small COMMON_VERBS list will incorrectly
    get a copula inserted before it.
  - Only ONE WH-word and ONE negation are handled per phrase; multiple
    of either will not compose correctly.
  - This only runs on SIGN-modality turns. Speech (Whisper) text is
    already natural language and passes through unchanged.
"""

from __future__ import annotations

import re

from . import lexicon


_PAREN_RE = re.compile(r"\s*\([^)]*\)")


def _strip_dictionary_annotations(words: list[str]) -> list[str]:
    """Many RealSASL vocabulary entries include parenthetical
    disambiguation notes meant for the dictionary, not for display --
    e.g. 'APPLE (1ST VARIANT)', 'ABOUT (TIME)', 'AMERICA (2ND VARIANT)'.
    ~11.5% of the real vocabulary has these. Strip them before doing
    anything else, or they leak straight into the output sentence."""
    cleaned = []
    for w in words:
        w = _PAREN_RE.sub("", w).strip()
        if w:
            cleaned.append(w)
    return cleaned


def _fix_possessives(words: list[str]) -> list[str]:
    """Pronoun immediately followed by a possessable noun -> possessive form."""
    out = list(words)
    for i in range(len(out) - 1):
        if out[i] in lexicon.PRONOUNS and out[i + 1] in lexicon.POSSESSABLE_NOUNS:
            out[i] = lexicon.POSSESSIVE_FORM[out[i]]
    return out


def _find_pronoun(words: list[str]) -> tuple[int, str] | tuple[None, None]:
    """Returns (index, original_pronoun_token) for the first bare
    (not-yet-converted) pronoun found, or (None, None)."""
    for i, w in enumerate(words):
        if w in lexicon.PRONOUNS:
            return i, w
    return None, None


def _lower_display(word: str) -> str:
    """Words already converted to their natural-language form (possessives,
    subject pronouns) are lowercase strings; anything still an
    all-caps gloss just gets lowercased for display."""
    return word if not word.isupper() else word.lower()


def _build_question(words: list[str]) -> str:
    words = _fix_possessives(words)

    wh_idx = next((i for i, w in enumerate(words) if w in lexicon.WH_WORDS), None)
    if wh_idx is None:
        # Tagged as a question but no WH-word found (e.g. a yes/no
        # question like "YOU HAPPY?") -- just punctuate as a question
        # rather than trying to invert word order without an
        # auxiliary-verb model.
        return _build_statement(words, force_punct="?")

    wh_word = words[wh_idx]
    remaining = words[:wh_idx] + words[wh_idx + 1:]

    pronoun_idx, pronoun = _find_pronoun(remaining)
    copula = lexicon.COPULA_BY_PRONOUN.get(pronoun, "is")

    if pronoun_idx is not None:
        remaining[pronoun_idx] = lexicon.SUBJECT_FORM[pronoun]

    rest = " ".join(_lower_display(w) for w in remaining)
    if rest:
        sentence = f"{wh_word.capitalize()} {copula} {rest}?"
    else:
        sentence = f"{wh_word.capitalize()}?"
    return sentence


def _build_statement(words: list[str], intent: str = "statement",
                      force_punct: str = None) -> str:
    words = _fix_possessives(words)

    negated = False
    if any(w in lexicon.NEGATION_WORDS for w in words):
        negated = True
        words = [w for w in words if w not in lexicon.NEGATION_WORDS]

    pronoun_idx, pronoun = _find_pronoun(words)
    out_words = list(words)

    if pronoun_idx is not None:
        subject = lexicon.SUBJECT_FORM[pronoun]
        out_words[pronoun_idx] = subject
        next_idx = pronoun_idx + 1
        next_is_verb = (
            next_idx < len(out_words) and out_words[next_idx] in lexicon.COMMON_VERBS
        )
        if not next_is_verb:
            copula = lexicon.COPULA_BY_PRONOUN[pronoun]
            if negated:
                copula += " not"
                negated = False  # already applied
            out_words.insert(next_idx, copula)
        elif negated:
            out_words.insert(next_idx, "do not" if pronoun != "HE" and pronoun != "SHE" and pronoun != "IT"
                              else "does not")
            negated = False

    text = " ".join(_lower_display(w) for w in out_words)
    if negated:
        # No pronoun/copula slot found to attach "not" to -- fall back
        # to a simple prefix rather than guessing at sentence structure.
        text = f"No, {text}" if text else "No."

    if not text:
        return ""

    text = text[0].upper() + text[1:]

    if force_punct:
        return text + force_punct
    if intent in lexicon.GREETING_INTENTS_FOR_EXCLAMATION:
        return text + "!"
    return text + "."


def build_sentence(words: list[str], intent: str = "statement") -> str:
    """
    Main entry point. Takes the raw recognized glosses for one
    finalized phrase (already in signed order) plus its tagged intent,
    and returns a best-effort natural-English rendering.

    words: list of gloss strings, e.g. ["YOU", "NAME", "WHAT"]
    intent: one of conversation/intent.py's categories
            (greeting, farewell, thanks, question, affirmation,
            negation, statement)
    """
    if not words:
        return ""

    words = _strip_dictionary_annotations(words)
    if not words:
        return ""

    has_wh = any(w in lexicon.WH_WORDS for w in words)
    if intent == "question" or has_wh:
        return _build_question(words)
    return _build_statement(words, intent=intent)
