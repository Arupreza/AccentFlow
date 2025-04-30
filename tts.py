import streamlit as st
import os
import torch
import tempfile
import time
from transformers import pipeline, AutoTokenizer
from pydub import AudioSegment
import re
import difflib
from TTS.api import TTS
import base64
import numpy as np
from dotenv import load_dotenv

# Function to add log messages
def add_log(message, type="info"):
    st.session_state.log_messages.append((message, type))
    if type == "error":
        st.error(message)

# TTS Model
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
    

def trim_reference(input_path, output_path, seconds=5):
    """Trim a reference audio file to the specified length."""
    try:
        if not os.path.exists(input_path):
            add_log(f"Reference audio file not found: {input_path}", "error")
            return False
            
        add_log(f"Trimming reference audio to {seconds} seconds", "info")
        audio = AudioSegment.from_file(input_path)
        
        # Check if audio is long enough
        if len(audio) < seconds * 1000:
            add_log(f"Warning: Audio file is shorter than {seconds} seconds. Using entire file.", "warning")
            trimmed_audio = audio
        else:
            trimmed_audio = audio[:seconds * 1000]
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        trimmed_audio.export(output_path, format="wav")
        st.session_state.generated_files["trimmed_ref"] = output_path
        add_log("✅ Reference audio trimmed successfully", "success")
        return True
    except Exception as e:
        add_log(f"❌ Error trimming reference audio: {e}", "error")
        return False
    

def generate_tts(text, reference_path, output_path, tts_model):
    """Synthesize text using XTTS-v2 with reference audio."""
    try:
        if not os.path.exists(reference_path):
            add_log(f"❌ Reference audio not found at: {reference_path}", "error")
            return False
            
        trimmed_ref_path = os.path.join(os.path.dirname(reference_path), "trimmed_ref.wav")
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(trimmed_ref_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Trim reference audio
        if not trim_reference(reference_path, trimmed_ref_path, seconds=5):
            add_log("Failed to trim reference audio", "error")
            return False

        # Check if model is loaded
        if tts_model is None:
            add_log("TTS model not loaded", "error")
            return False

        # Synthesize
        add_log("Generating speech from text (this may take a while)...", "info")
        with st.spinner("Generating speech from text..."):
            tts_model.tts_to_file(
                text=text,
                speaker_wav=[trimmed_ref_path],
                language="en",
                file_path=output_path
            )
            
        st.session_state.generated_files["tts_output"] = output_path
        add_log("✅ TTS synthesis complete", "success")
        return True
    except Exception as e:
        add_log(f"❌ Error generating TTS: {e}", "error")
        return False