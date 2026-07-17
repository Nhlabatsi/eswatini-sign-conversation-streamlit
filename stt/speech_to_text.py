"""
speech_to_text.py
==================
Wraps OpenAI's Whisper for the "Hearing person speaks -> text" stage
of the pipeline (Automatic Speech Recognition).
    Microphone / audio file -> Whisper -> text
Whisper is used here (rather than a from-scratch model) because
speech-to-text for major spoken languages is a well-solved problem
with strong open pretrained models -- the actually novel, hard part of
this project is the sign-language side, which is exactly where custom
work (the TCN/ArcFace pipeline) belongs. No point reinventing ASR.
"""

from __future__ import annotations

import numpy as np

try:
    import whisper
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "openai-whisper is required: pip install openai-whisper\n"
        "Also requires ffmpeg on PATH: apt-get install -y ffmpeg (Colab/Linux) "
        "or brew install ffmpeg (Mac)."
    ) from e


class SpeechToText:
    """
    Thin wrapper around a Whisper model.
    model_size options (speed/accuracy tradeoff, smallest to largest):
        "tiny", "base", "small", "medium", "large-v3"
    "base" is a reasonable default for a first working pipeline;
    "small" or "medium" will be notably more accurate if you have the
    compute budget (especially for a language/accent Whisper wasn't
    heavily trained on -- worth checking transcription quality for
    Eswatini/siSwati-accented English or code-switching speech, which
    may need a larger model than you'd guess from English-only demos).
    """

    def __init__(self, model_size: str = "base", device: str | None = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)
        self.model_size = model_size

    def transcribe_file(self, audio_path: str, language: str | None = None) -> dict:
        """
        Transcribe an audio file (wav/mp3/m4a/etc -- anything ffmpeg can read).
        Args:
            audio_path: path to the audio file.
            language: optional ISO language code (e.g. "en") to force;
                      if None, Whisper auto-detects.
        Returns:
            dict with keys: "text" (full transcript), "language"
            (detected or forced language), "segments" (list of
            per-segment dicts with start/end/text, useful if you later
            want to align speech timing with sign display timing).
        """
        result = self.model.transcribe(audio_path, language=language)
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
        }

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000,
                          language: str | None = None) -> dict:
        """
        Transcribe raw audio already in memory (e.g. from a microphone
        buffer), rather than a file on disk.
        Args:
            audio: float32 mono numpy array, values roughly in [-1, 1].
            sample_rate: MUST be 16000 (Whisper's expected rate) -- resample
                         before calling this if your mic captures at a
                         different rate.
            language: optional ISO language code to force.
        """
        if sample_rate != 16000:
            raise ValueError(
                f"Whisper expects 16000 Hz audio, got sample_rate={sample_rate}. "
                f"Resample first (e.g. with librosa.resample or scipy)."
            )
        audio = audio.astype(np.float32)
        result = self.model.transcribe(audio, language=language)
        return {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                for s in result.get("segments", [])
            ],
        }
