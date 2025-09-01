import os
import torch
import streamlit as st
from transformers import pipeline, AutoTokenizer
from TTS.api import TTS
from .ui import add_log

@st.cache_resource
def load_grammar_model():
    try:
        with st.spinner("Loading grammar correction model..."):
            HF_TOKEN = os.getenv("HF_TOKEN")
            model_name = "vennify/t5-base-grammar-correction"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
            model = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)
            add_log("✅ Grammar correction model loaded successfully", "success")
            return model
    except Exception as e:
        add_log(f"❌ Error loading grammar model: {e}", "error")
        return None

@st.cache_resource
def load_tts_model():
    try:
        with st.spinner("Loading TTS model..."):
            use_gpu = torch.cuda.is_available()
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
            add_log(f"✅ TTS model loaded successfully (GPU: {use_gpu})", "success")
            return tts
    except Exception as e:
        add_log(f"❌ Error loading TTS model: {e}", "error")
        return None