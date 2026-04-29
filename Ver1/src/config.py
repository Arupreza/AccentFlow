# Core tunables
SEGMENT_DURATION = 10        # seconds for ASR chunking
TTS_MAX_CHARS     = 300      # sentence-aware cap for XTTS chunking
SILENCE_PAD_MS    = 30       # small gap between aligned chunks to avoid clicks

# Language support (English, Korean, Indonesian)
LANG_TO_XTTS = {
    "en": "en",  # English
    "ko": "ko",  # Korean
    "id": "id",  # Indonesian (Bahasa Indonesia)
}
DEFAULT_TTS_LANG = "en"

# UI choices (Process tab)
SUPPORTED_LANGS = [
    ("auto", "Auto-detect"),
    ("en",   "English"),
    ("ko",   "Korean"),
    ("id",   "Indonesian"),
]