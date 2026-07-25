"""
grammar: the Grammar & Sentence Builder stage (Stage 2).

Takes raw recognized sign glosses (in signed order) plus a tagged
intent, and produces a best-effort natural-English rendering. See
builder.py's module docstring for exactly what patterns are handled
and, importantly, what's explicitly NOT handled yet.
"""
from .builder import build_sentence

__all__ = ["build_sentence"]
