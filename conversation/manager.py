"""
manager.py
===========
ConversationManager: the orchestration hub sitting between raw
recognition output (SLR word-by-word, STT utterance-by-utterance) and
the shared conversation transcript.

Stage 1 of the improved architecture:

    Sign Recognition (SLR) --\\
                               >--> Conversation Manager --> transcript
    Speech Recognition (STT)-/           |
                                          |-- Phrase Builder (buffers SLR words into phrases)
                                          |-- Context & Memory (turn history)
                                          |-- Intent & Topics (lightweight tagging)

Stage 2 (see conversation/grammar/ and conversation/gloss_to_text.py):
finalized sign phrases are now run through a Grammar & Sentence Builder,
producing a "sentence" field alongside the raw "text" (recognized
glosses) -- e.g. text="YOU NAME WHAT", sentence="What is your name?".
Speech turns already contain natural language (Whisper output), so
their "sentence" field is just the same text unchanged.

    As of this revision, the sentence builder is LM-based: it calls
    SmolLM2-135M-Instruct (conversation/gloss_to_text.py) to turn the
    raw gloss into a fluent sentence, since a small general-purpose
    instruct model handles the "clean up this rough note into natural
    English" task well without needing ESL-specific training data --
    the sign side still does all the actual visual-language learning.
    The original rule-based build_sentence() (conversation/grammar/)
    is kept as an automatic fallback: if the LM fails to load or
    errors at inference time (e.g. no network to fetch the model
    weights, out-of-memory on a constrained host), we fall back to the
    rule-based sentence rather than losing the "sentence" field or
    crashing the conversation aid.

NOT yet implemented (later stage, by design):
    - Text-to-Speech output

Usage (see streamlit_app.py for the actual integration):

    manager = ConversationManager()
    manager.add_sign_word("HELLO", confidence=0.91)
    ...
    manager.maybe_finalize_sign_phrase()   # call periodically (e.g. every fragment tick)
    manager.add_speech_text("Hi there, how are you?")
    for turn in manager.get_transcript():
        ...

THREADING NOTE: like the rest of this app, ConversationManager is NOT
thread-safe and is not meant to be touched from the WebRTC video
callback thread directly. The existing pattern still applies: the
callback thread writes into its own locked LiveSignState, and the main
Streamlit script (e.g. inside the polling fragment) is what calls into
ConversationManager -- keeping all conversation-state mutation on a
single thread.
"""

from __future__ import annotations

import logging
import time

from .phrase_builder import PhraseBuilder
from .context import ConversationContext
from . import intent as intent_module
from .grammar import build_sentence
from .gloss_to_text import GlossToTextConverter

SIGN_SPEAKER = "Deaf person"
SPEECH_SPEAKER = "Hearing person"

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, phrase_pause_seconds: float = 3.0, phrase_max_words: int = 8):
        self._phrase_pause_seconds = phrase_pause_seconds
        self._phrase_max_words = phrase_max_words
        self.sign_phrase_builder = PhraseBuilder(
            pause_seconds=phrase_pause_seconds, max_words=phrase_max_words
        )
        self.context = ConversationContext()

    # -- Sign side: word-by-word input, buffered into phrases ------------

    def add_sign_word(self, word: str, confidence: float = 1.0, timestamp: float = None):
        """Feed one newly-recognized sign word into the pending phrase
        buffer. Does NOT finalize a turn by itself -- call
        maybe_finalize_sign_phrase() (e.g. on every periodic poll) to
        check whether the phrase is done."""
        self.sign_phrase_builder.add_word(word, confidence, timestamp)

    def pending_sign_phrase_preview(self) -> str:
        """Words signed so far in the current, not-yet-finalized phrase
        -- useful for showing "currently signing: ..." live feedback
        before the phrase is committed to the transcript."""
        return " ".join(self.sign_phrase_builder.words)

    def maybe_finalize_sign_phrase(self, now: float = None) -> dict | None:
        """Call periodically (e.g. every ~1s poll). If enough time has
        passed since the last sign word (or the buffer capped out),
        finalizes the buffered phrase into a turn and returns it;
        otherwise returns None."""
        now = now if now is not None else time.time()
        if not self.sign_phrase_builder.should_finalize(now):
            return None
        return self._finalize_sign_phrase()

    def force_finalize_sign_phrase(self) -> dict | None:
        """Finalize whatever's pending right now, regardless of pause
        timing -- e.g. call this when the hearing person starts
        speaking, so the sign phrase doesn't linger half-finished."""
        if not self.sign_phrase_builder.has_pending():
            return None
        return self._finalize_sign_phrase()

    def _finalize_sign_phrase(self) -> dict | None:
        phrase = self.sign_phrase_builder.finalize()
        if phrase is None:
            return None
        tagged_intent, topics = intent_module.classify(phrase["text"])
        sentence = self._build_sentence(phrase["text"], phrase["words"], tagged_intent)
        return self.context.add_turn(
            speaker=SIGN_SPEAKER, modality="sign", text=phrase["text"],
            sentence=sentence, intent=tagged_intent, topics=topics,
            timestamp=phrase["end_time"],
        )

    def _build_sentence(self, gloss_text: str, words, tagged_intent) -> str:
        """Turn a finalized gloss ('YOU NAME WHAT') into a natural
        English sentence ('What is your name?').

        Tries the LM-based converter (SmolLM2-135M-Instruct) first.
        Falls back to the original rule-based build_sentence() if the
        LM raises for any reason -- model not downloaded yet, no
        network, OOM, etc. A degraded rule-based sentence is far
        better than crashing a live two-way conversation aid.
        """
        try:
            return GlossToTextConverter.get().convert(gloss_text)
        except Exception:
            logger.exception(
                "GlossToTextConverter failed on gloss %r -- falling back "
                "to rule-based build_sentence()", gloss_text
            )
            return build_sentence(words, tagged_intent)

    # -- Speech side: Whisper already gives full utterances ---------------

    def add_speech_text(self, text: str, timestamp: float = None) -> dict | None:
        """Speech-to-text utterances are already phrase/sentence-sized
        (unlike individual sign glosses), so this finalizes a turn
        immediately rather than buffering."""
        text = (text or "").strip()
        if not text:
            return None
        # If the Deaf person had a sign phrase mid-buffer when the
        # hearing person started speaking, finalize it first so turns
        # stay in the right chronological order rather than getting
        # stuck pending indefinitely.
        self.force_finalize_sign_phrase()
        tagged_intent, topics = intent_module.classify(text)
        return self.context.add_turn(
            speaker=SPEECH_SPEAKER, modality="speech", text=text,
            intent=tagged_intent, topics=topics, timestamp=timestamp,
        )

    # -- Access for rendering ----------------------------------------------

    def get_transcript(self) -> list[dict]:
        return self.context.turns

    def reset(self):
        self.sign_phrase_builder = PhraseBuilder(
            pause_seconds=self._phrase_pause_seconds,
            max_words=self._phrase_max_words,
        )
        self.context = ConversationContext()
