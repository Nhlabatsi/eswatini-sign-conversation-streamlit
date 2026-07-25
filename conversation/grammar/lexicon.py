"""
lexicon.py
===========
Small hand-built word lists driving the rule-based grammar builder.
Nothing here is a trained model -- it's a deliberately narrow set of
patterns covering common cases, documented honestly so it's clear what
it does and doesn't handle (see builder.py's module docstring for the
full list of known limitations).
"""

PRONOUNS = {"I", "ME", "YOU", "HE", "SHE", "WE", "THEY", "IT"}

# Subject-position form (how the pronoun reads when it's the grammatical subject).
SUBJECT_FORM = {
    "I": "I", "ME": "I", "YOU": "you", "HE": "he", "SHE": "she",
    "WE": "we", "THEY": "they", "IT": "it",
}

# Possessive form (how the pronoun reads when it modifies a noun, e.g. "your name").
POSSESSIVE_FORM = {
    "I": "my", "ME": "my", "YOU": "your", "HE": "his", "SHE": "her",
    "WE": "our", "THEY": "their", "IT": "its",
}

# Copula ("to be") to use when this pronoun is the subject.
COPULA_BY_PRONOUN = {
    "I": "am", "ME": "am", "YOU": "are", "WE": "are", "THEY": "are",
    "HE": "is", "SHE": "is", "IT": "is",
}

WH_WORDS = {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"}

NEGATION_WORDS = {"NOT", "NO", "NEVER"}

# Nouns commonly following a pronoun where the pronoun should read as
# possessive ("YOU NAME" -> "your name") rather than as a subject
# ("YOU NAME" -> "you name", which doesn't make sense as a statement).
# Deliberately small and hand-picked -- extend as real usage reveals
# more common patterns, rather than trying to guess exhaustively.
POSSESSABLE_NOUNS = {
    "NAME", "HOUSE", "FAMILY", "FRIEND", "MOTHER", "FATHER", "SISTER",
    "BROTHER", "JOB", "AGE", "BIRTHDAY", "PHONE", "CAR", "SCHOOL",
    "TEACHER", "DOG", "CAT", "BOOK", "BAG", "MONEY", "ROOM",
}

# Common verb glosses -- if a pronoun is immediately followed by one of
# these, it's already a verb and should NOT get a copula inserted
# before it (e.g. "YOU GO" stays "you go", not "you are go"). Any verb
# not in this small list will incorrectly get a copula inserted --
# a known, documented limitation rather than a silent guess.
COMMON_VERBS = {
    "GO", "COME", "RUN", "WALK", "EAT", "DRINK", "WANT", "LIKE", "LOVE",
    "WORK", "PLAY", "STUDY", "LEARN", "TEACH", "HELP", "SEE", "LOOK",
    "KNOW", "THINK", "FEEL", "NEED", "HAVE", "MAKE", "GIVE", "TAKE",
    "BUY", "SELL", "READ", "WRITE", "SIGN", "SPEAK", "LISTEN", "HEAR",
    "UNDERSTAND", "REMEMBER", "FORGET", "FINISH", "START", "STOP",
    "LIVE", "STAY", "MOVE", "SIT", "STAND", "SLEEP", "WAKE", "DRIVE",
}

GREETING_INTENTS_FOR_EXCLAMATION = {"greeting", "farewell", "thanks"}
