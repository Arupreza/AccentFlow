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

# Load Grammar Correction Model
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
    
# Core functions
def correct_text(text, grammar_model):
    """Use the grammar correction model directly."""
    try:
        if not text.strip():
            return text
        
        if grammar_model is None:
            add_log("Grammar model not loaded. Returning original text.", "warning")
            return text
            
        return grammar_model(text)[0]['generated_text']
    except Exception as e:
        add_log(f"Error correcting text: {e}", "error")
        return text  # Return original text on error

def remove_redundant_text(text):
    """Removes redundant occurrences of 'I :' in the corrected text."""
    return re.sub(r'\bI :\s*', '', text)

def transcribe_video(video_path):
    """Transcribe full text from video using OpenAI Whisper model with timestamps."""
    try:
        if not os.path.exists(video_path):
            add_log(f"Video file not found: {video_path}", "error")
            return "", []
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        add_log(f"Using device: {device} for transcription", "info")
        
        with st.spinner("Transcribing video (this may take a while)..."):
            asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-medium", 
                device=0 if device == "cuda" else -1,
                return_timestamps=True
            )
            result = asr_pipeline(video_path)
            add_log("✅ Transcription complete", "success")
            return result["text"], result["chunks"]
    except Exception as e:
        add_log(f"❌ Error transcribing video: {e}", "error")
        return "", []  # Return empty results on error

def format_srt(transcription_chunks):
    """Format transcription into SRT format."""
    srt_content = []
    for idx, chunk in enumerate(transcription_chunks, start=1):
        try:
            start_time = chunk["timestamp"][0]
            end_time = chunk["timestamp"][1]
            text = chunk["text"].strip()

            def format_time(seconds):
                millisec = int((seconds % 1) * 1000)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02}:{minutes:02}:{secs:02},{millisec:03}"

            start_srt = format_time(start_time)
            end_srt = format_time(end_time)

            srt_content.append(f"{idx}\n{start_srt} --> {end_srt}\n{text}\n")
        except Exception as e:
            add_log(f"Error formatting chunk {idx}: {e}", "error")
            continue

    return "\n".join(srt_content)

def extract_audio_segments(video_path, output_dir, segment_duration):
    """Extract audio segments from video."""
    try:
        if not os.path.exists(video_path):
            add_log(f"Video file not found: {video_path}", "error")
            return None
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        add_log(f"Extracting audio from video", "info")
        full_audio = AudioSegment.from_file(video_path)
        
        full_wav_path = os.path.join(output_dir, "full_audio.wav")
        
        # Save the full audio first
        add_log(f"Saving full audio file", "info")
        full_audio.export(full_wav_path, format="wav")
        st.session_state.generated_files["full_audio"] = full_wav_path

        # Convert segment_duration from seconds to milliseconds
        segment_duration_ms = segment_duration * 1000

        # Calculate the number of segments
        segment_count = len(full_audio) // segment_duration_ms
        add_log(f"Creating {segment_count} audio segments", "info")

        for i in range(segment_count):
            start_time = i * segment_duration_ms
            end_time = (i + 1) * segment_duration_ms
            
            segment = full_audio[start_time:end_time]
            segment_path = os.path.join(output_dir, f"segment_{i+1}.wav")
            segment.export(segment_path, format="wav")
            st.session_state.generated_files[f"segment_{i+1}"] = segment_path

        add_log("✅ Audio extraction complete", "success")
        return full_wav_path
    except Exception as e:
        add_log(f"❌ Error extracting audio: {e}", "error")
        return None
    

def correct_grammar_in_subtitles(subtitle_file, incorrect_paragraph_file, corrected_paragraph_file, grammar_model):
    """Read subtitles, generate incorrect and corrected paragraphs."""
    try:
        if not os.path.exists(subtitle_file):
            add_log(f"Subtitle file not found: {subtitle_file}", "error")
            return False
            
        add_log(f"Reading subtitles and correcting grammar", "info")
        with open(subtitle_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        incorrect_text = []
        corrected_text = []
        
        for line in lines:
            # Only process lines that are not timestamps or index numbers and are not empty
            if "-->" not in line and not line.strip().isdigit() and line.strip():
                incorrect_text.append(line.strip())
                corrected_text_line = correct_text(line.strip(), grammar_model)
                cleaned_text_line = remove_redundant_text(corrected_text_line)
                corrected_text.append(cleaned_text_line)

        # Ensure output directories exist
        os.makedirs(os.path.dirname(incorrect_paragraph_file), exist_ok=True)
        os.makedirs(os.path.dirname(corrected_paragraph_file), exist_ok=True)

        incorrect_para = " ".join(incorrect_text)
        corrected_para = " ".join(corrected_text)
        
        with open(incorrect_paragraph_file, "w", encoding="utf-8") as file:
            file.write(incorrect_para)
            
        with open(corrected_paragraph_file, "w", encoding="utf-8") as file:
            file.write(corrected_para)
            
        st.session_state.generated_files["incorrect_text"] = incorrect_paragraph_file
        st.session_state.generated_files["corrected_text"] = corrected_paragraph_file
        
        add_log("✅ Grammar correction complete", "success")
        return True, incorrect_para, corrected_para
    except Exception as e:
        add_log(f"❌ Error correcting grammar: {e}", "error")
        return False, "", ""

def highlight_corrections(original, corrected):
    """Highlight corrections in the corrected paragraph using HTML formatting."""
    try:
        diff = difflib.ndiff(original.split(), corrected.split())
        highlighted_text = " ".join(
            [f"<span style='color: red; font-weight: bold;'>{word[2:]}</span>" if word.startswith("+ ") else word[2:]
            for word in diff if not word.startswith("- ")]
        )
        return highlighted_text
    except Exception as e:
        add_log(f"Error highlighting corrections: {e}", "error")
        return corrected  # Return unformatted text on error