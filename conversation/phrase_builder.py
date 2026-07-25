"""
phrase_builder.py
==================
Accumulates a stream of individually-recognized sign words (glosses)
into phrases, since our SLR model recognizes one isolated sign per
clip/window rather than full sentences. A phrase is finalized when:
  - enough time has passed since the last word (a pause), or
  - the buffer grows past a safety cap, or
  - it's explicitly force-finalized (e.g. the other party starts talking).

NOTE: this stage only *groups* words -- it does not reorder them into
grammatical English. That's the explicit job of a later stage (Grammar
& Sentence Builder), not implemented yet. For now, a "phrase" is just
the recognized glosses joined in the order they were signed.
"""

from __future__ import annotations

import time


class PhraseBuilder:
    def __init__(self, pause_seconds: float = 3.0, max_words: int = 8):
        self.pause_seconds = pause_seconds
        self.max_words = max_words
        self.words: list[str] = []
        self.word_confidences: list[float] = []
        self.last_word_time: float = 0.0
        self.phrase_start_time: float | None = None

    def add_word(self, word: str, confidence: float = 1.0, timestamp: float = None):
        now = timestamp if timestamp is not None else time.time()
        if self.phrase_start_time is None:
            self.phrase_start_time = now
        self.words.append(word)
        self.word_confidences.append(confidence)
        self.last_word_time = now

    def has_pending(self) -> bool:
        return len(self.words) > 0

    def should_finalize(self, now: float = None) -> bool:
        if not self.has_pending():
            return False
        now = now if now is not None else time.time()
        if len(self.words) >= self.max_words:
            return True
        return (now - self.last_word_time) >= self.pause_seconds

    def finalize(self) -> dict | None:
        """Returns a dict describing the finalized phrase, or None if
        nothing was pending. Clears the buffer either way."""
        if not self.has_pending():
            return None
        phrase = {
            "text": " ".join(self.words),
            "words": list(self.words),
            "avg_confidence": sum(self.word_confidences) / len(self.word_confidences),
            "start_time": self.phrase_start_time,
            "end_time": self.last_word_time,
        }
        self.words = []
        self.word_confidences = []
        self.phrase_start_time = None
        return phrase
