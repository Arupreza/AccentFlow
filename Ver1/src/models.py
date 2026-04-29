import os
import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer
from TTS.api import TTS
from .ui import add_log

@st.cache_resource
def load_grammar_model():
    try:
        with st.spinner("Loading grammar model…"):
            model_name = "vennify/t5-base-grammar-correction"
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN"))
            mdl = pipeline("text2text-generation", model=model_name, tokenizer=tok)
            add_log("✅ Grammar model ready", "success")
            return mdl
    except Exception as e:
        add_log(f"❌ Grammar model error: {e}", "error")
        return None

@st.cache_resource
def load_tts_model():
    try:
        with st.spinner("Loading TTS…"):
            use_gpu = torch.cuda.is_available()
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
            add_log(f"✅ TTS ready (GPU={use_gpu})", "success")
            return tts
    except Exception as e:
        add_log(f"❌ TTS load error: {e}", "error")
        return None