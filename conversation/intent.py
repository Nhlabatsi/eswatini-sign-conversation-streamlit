"""
intent.py
==========
Lightweight, rule-based intent and topic tagging for a finalized
phrase/utterance. Deliberately simple for this stage -- a fixed set of
keyword/pattern rules rather than a trained classifier, which is a
reasonable scope for "tag this utterance well enough for the
Conversation Manager to make basic decisions with," and can be swapped
for a real model later without changing the Conversation Manager's
interface (classify() keeps the same signature either way).
"""

from __future__ import annotations

import re

_GREETING_WORDS = {"hello", "hi", "hey"}
_FAREWELL_WORDS = {"bye", "goodbye", "farewell"}
_THANKS_WORDS = {"thank", "thanks"}
_AFFIRM_WORDS = {"yes", "yeah", "ok", "okay", "sure"}
_NEGATE_WORDS = {"no", "not", "never"}
_WH_WORDS = {"what", "who", "where", "when", "why", "how", "which"}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "i", "you", "he", "she",
    "it", "we", "they", "to", "of", "and", "or", "in", "on", "at", "for",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def classify(text: str) -> tuple[str, list[str]]:
    """Returns (intent, topics) for a finalized phrase/utterance.

    intent is one of: greeting, farewell, thanks, question, affirmation,
    negation, statement.
    topics is a rough list of non-stopword content tokens -- not real
    topic modeling, just enough to give downstream stages something to
    work with until this is upgraded.
    """
    tokens = _tokenize(text)
    token_set = set(tokens)

    if token_set & _GREETING_WORDS:
        intent = "greeting"
    elif token_set & _FAREWELL_WORDS:
        intent = "farewell"
    elif token_set & _THANKS_WORDS:
        intent = "thanks"
    elif text.strip().endswith("?") or (token_set & _WH_WORDS):
        # WH-word checked anywhere in the phrase, not just first position:
        # sign-language phrases commonly put the WH-word LAST ("YOU NAME
        # WHAT"), unlike English's WH-initial convention. Checking only
        # tokens[0] (the original approach) would silently misclassify
        # exactly the phrases this system exists to handle correctly.
        intent = "question"
    elif token_set & _AFFIRM_WORDS:
        intent = "affirmation"
    elif token_set & _NEGATE_WORDS:
        intent = "negation"
    else:
        intent = "statement"

    topics = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return intent, topics
