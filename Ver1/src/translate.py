from __future__ import annotations
import streamlit as st
import torch
from transformers import pipeline
from .ui import add_log

# Map direct translation models (Helsinki) for our three languages
# We pivot via English for ko<->id (ko->en->id, id->en->ko)
_MODEL_FOR = {
    ("en","ko"): "Helsinki-NLP/opus-mt-en-ko",
    ("ko","en"): "Helsinki-NLP/opus-mt-ko-en",
    ("en","id"): "Helsinki-NLP/opus-mt-en-id",
    ("id","en"): "Helsinki-NLP/opus-mt-id-en",
}

@st.cache_resource
def _get_translator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("translation", model=model_name, device=device)

def _translate_once(text: str, src: str, tgt: str) -> str:
    if src == tgt or not text.strip():
        return text
    key = (src, tgt)
    if key not in _MODEL_FOR:
        raise ValueError(f"No direct model for {src}->{tgt}")
    trans = _get_translator(_MODEL_FOR[key])
    # chunk roughly by 800-1000 chars to be safe
    chunks, cur = [], ""
    for sent in text.split(". "):
        s = (sent.strip() + ("" if sent.endswith(".") else ".")).strip()
        if len(cur) + len(s) < 900:
            cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)

    outs = []
    for c in chunks:
        r = trans(c, max_length=1024)
        outs.append(r[0]["translation_text"])
    return " ".join(outs).strip()

def translate_text(text: str, src: str, tgt: str) -> str:
    """
    Translate text between en/ko/id. For ko<->id, pivot via English.
    """
    if src == tgt:
        return text
    try:
        if (src, tgt) in _MODEL_FOR:
            return _translate_once(text, src, tgt)
        # pivot via English
        if src == "ko" and tgt == "id":
            return _translate_once(_translate_once(text, "ko","en"), "en","id")
        if src == "id" and tgt == "ko":
            return _translate_once(_translate_once(text, "id","en"), "en","ko")
        # default: no-op
        return text
    except Exception as e:
        add_log(f"Translation failed ({src}->{tgt}): {e}", "error")
        return text