"""
conversation/gloss_to_text.py

Stage 2 of the improved architecture: Grammar & Sentence Builder.

Takes a buffered ESL gloss string (e.g. "MY NAME VELI", "YOU EAT FINISH
QUESTION") -- the kind of thing the Conversation Manager's phrase buffer
already assembles from individually recognized sign words -- and asks a
small local language model (SmolLM2-135M-Instruct) to render it as
natural, fluent English.

WHY THIS WORKS WITH SO LITTLE DATA:
    The sign recognizer (SLR) stays exactly as it is -- it only ever needs
    to learn the *visual* language (hand shapes, motion, sequences). This
    module never sees video or landmarks; it only sees text. So instead of
    needing thousands of (video, natural-English) pairs to teach a model
    ESL end-to-end, we only need to teach a tiny LM a much easier, purely
    textual skill: "given rough gloss notes, write the fluent sentence a
    person would actually say." That's close to the kind of grammar-
    repair / note-expansion task general-purpose instruct models are
    already decent at out of the box, and it's cheap to improve later with
    a small number of (gloss, sentence) fine-tuning examples if needed --
    far less data than an end-to-end video-to-English model would require.

WHAT THIS STAGE DOES NOT DO (deferred):
    - It does not touch the sign model, the phrase-buffering/pause logic,
      or intent/topic tagging -- those stay in conversation/manager.py.
    - It does not do speech-side grammar correction (Whisper's STT output
      is already natural English).
    - It is not fine-tuned; it's a prompted, few-shot, greedy-decoded call
      to the stock SmolLM2-135M-Instruct checkpoint. Swap in a fine-tuned
      checkpoint later without changing the calling code.
"""

import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "You are helping translate Eswatini Sign Language (ESL) gloss into natural, "
    "fluent English. ESL gloss is a sequence of sign words in ESL word order, "
    "often without articles, tense markers, or plurals -- similar to a rough "
    "note. Rewrite the gloss as a single natural English sentence that a fluent "
    "speaker would say, preserving the original meaning exactly. Do not add "
    "information that isn't in the gloss. If the gloss ends with a question "
    "marker, phrase the output as a natural question. Output ONLY the "
    "translated sentence -- no explanation, no quotes, nothing else."
)

# A handful of illustrative examples is enough steering for a model this
# size; expand this list (or replace it with real ESL gloss samples) if you
# see systematic mistakes on your own data.
FEW_SHOT = [
    ("MY NAME VELI", "My name is Veli."),
    ("YOU EAT FINISH QUESTION", "Have you eaten?"),
    ("TOMORROW I GO SCHOOL", "I'm going to school tomorrow."),
    ("WHERE BATHROOM QUESTION", "Where is the bathroom?"),
    ("I NOT UNDERSTAND", "I don't understand."),
]


class GlossToTextConverter:
    """
    Loads SmolLM2-135M-Instruct once and converts gloss strings to sentences.

    Use GlossToTextConverter.get() rather than the constructor directly --
    it's a lazily-created singleton guarded by a lock, mirroring how
    load_sign_model()/load_speech_model() are wrapped with @st.cache_resource
    in streamlit_app.py (same goal: don't load the model multiple times
    across Streamlit's rerun-on-every-interaction behavior, and don't race
    two reruns into loading it twice concurrently).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", max_new_tokens: int = 40):
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()

    @classmethod
    def get(cls, device: str = "cpu") -> "GlossToTextConverter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device=device)
            return cls._instance

    def _build_prompt(self, gloss: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for gloss_ex, text_ex in FEW_SHOT:
            messages.append({"role": "user", "content": gloss_ex})
            messages.append({"role": "assistant", "content": text_ex})
        messages.append({"role": "user", "content": gloss})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def convert(self, gloss: str) -> str:
        """Convert one gloss phrase (e.g. 'MY NAME VELI') to natural English."""
        gloss = gloss.strip()
        if not gloss:
            return gloss

        prompt = self._build_prompt(gloss)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # greedy -- this is a translation
                temperature=None,         # task, not creative writing; we
                top_p=None,               # want the most literal, stable
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._clean(text)

        # SmolLM2-135M is tiny and will occasionally rewrite in circles,
        # echo the prompt, or return nothing usable. Rather than surface
        # garbage in a conversation aid two people rely on to understand
        # each other, fall back to the raw gloss.
        return text if text else self._fallback(gloss)

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip().strip('"').strip()
        # Keep only the first sentence-ish chunk in case the model rambles
        # past the translation into extra commentary.
        pieces = re.split(r"(?<=[.!?])\s", text, maxsplit=1)
        text = pieces[0].strip() if pieces else text
        return text

    @staticmethod
    def _fallback(gloss: str) -> str:
        return gloss.capitalize()
