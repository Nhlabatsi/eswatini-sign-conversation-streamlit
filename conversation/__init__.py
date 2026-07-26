"""
conversation: the Conversation Manager stage of the pipeline.

    Sign Recognition (SLR) --\\
                               >--> Conversation Manager --> transcript
    Speech Recognition (STT)-/

See manager.py for the orchestrator and its docstring for what's
implemented at this stage (phrase buffering, turn history, lightweight
intent/topic tagging, and Stage 2's grammar-built sentences) vs.
deliberately deferred (Text-to-Speech).
"""
from .manager import ConversationManager, SIGN_SPEAKER, SPEECH_SPEAKER

__all__ = ["ConversationManager", "SIGN_SPEAKER", "SPEECH_SPEAKER"]
