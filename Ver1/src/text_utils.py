import os
import nltk
import streamlit as st
from .ui import add_log

def _fmt_time(t):
    ms = int((t % 1) * 1000); h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def format_srt(chunks):
    out = []
    for i, ch in enumerate(chunks, 1):
        try:
            s, e = ch["timestamp"]
            txt = ch["text"].strip()
            out.append(f"{i}\n{_fmt_time(s)} --> { _fmt_time(e)}\n{txt}\n")
        except Exception as err:
            add_log(f"SRT format error at {i}: {err}", "error")
    return "\n".join(out)

def _chunk_sentences(text: str, max_sents=3):
    sents = nltk.sent_tokenize(text)
    return [" ".join(sents[i:i+max_sents]) for i in range(0, len(sents), max_sents)]

def _correct_chunk(model, chunk: str):
    if not chunk.strip(): return chunk
    try:
        out = model(chunk, max_length=512, truncation=True)
        return out[0]["generated_text"]
    except Exception as e:
        add_log(f"Grammar chunk error: {e}", "error")
        return chunk

def english_grammar_correction(text: str, model):
    """Run GEC if model provided; otherwise return original."""
    if model is None: return text
    chunks = _chunk_sentences(text, 3)
    fixed = [ _correct_chunk(model, c) for c in chunks ]
    return " ".join(fixed)

def highlight_corrections(original: str, corrected: str) -> str:
    try:
        import difflib
        diff = difflib.ndiff(original.split(), corrected.split())
        return " ".join(
            [f"<b style='color:#d33'>{w[2:]}</b>" if w.startswith("+ ") else w[2:]
             for w in diff if not w.startswith("- ")]
        )
    except Exception:
        return corrected