"""
context.py
===========
Lightweight short-term conversation memory: a running list of
finalized turns (from either party/modality), with a few simple
derived stats (last speaker, turn counts) that later stages (Grammar &
Sentence Builder, and eventually a smarter Intent/Topic module) can
use for context -- e.g. "was the last turn a question, so this reply
is likely an answer".
"""

from __future__ import annotations

import time


class ConversationContext:
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.turns: list[dict] = []

    def add_turn(self, speaker: str, modality: str, text: str,
                 intent: str = None, topics: list[str] = None,
                 timestamp: float = None, sentence: str = None) -> dict:
        turn = {
            "speaker": speaker,       # e.g. "Deaf person" / "Hearing person"
            "modality": modality,     # "sign" or "speech"
            "text": text,             # raw recognized glosses / transcribed speech
            "sentence": sentence if sentence is not None else text,  # grammar-built rendering
            "intent": intent,
            "topics": topics or [],
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        self.turns.append(turn)
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history:]
        return turn

    @property
    def last_turn(self) -> dict | None:
        return self.turns[-1] if self.turns else None

    @property
    def last_speaker(self) -> str | None:
        return self.last_turn["speaker"] if self.last_turn else None

    def recent_turns(self, n: int = 5) -> list[dict]:
        return self.turns[-n:]

    def turn_count(self, speaker: str = None) -> int:
        if speaker is None:
            return len(self.turns)
        return sum(1 for t in self.turns if t["speaker"] == speaker)
