import os
import torch
import streamlit as st
from Assets_Text import transcribe_video, format_srt, extract_audio_segments, correct_grammar_in_subtitles, highlight_corrections

# Define paths dynamically
OUTPUT_DIR = "saved_output"
SUBTITLE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "transcribed_subtitles.srt")
INCORRECT_PARAGRAPH_FILE = os.path.join(OUTPUT_DIR, "incorrect_paragraph.txt")
CORRECTED_PARAGRAPH_FILE = os.path.join(OUTPUT_DIR, "corrected_paragraph.txt")
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extracted_audio_segments")
FULL_AUDIO_OUTPUT = os.path.join(OUTPUT_DIR, "full_audio.wav")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Streamlit App
st.title("Grammar Correction App")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    video_path = os.path.join(OUTPUT_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.success("Video uploaded successfully!")
    
    transcribed_text, transcription_chunks = transcribe_video(video_path)
    subtitle_text = format_srt(transcription_chunks)
    with open(SUBTITLE_OUTPUT_FILE, "w", encoding="utf-8") as srt_file:
        srt_file.write(subtitle_text)
    extract_audio_segments(video_path, transcription_chunks, AUDIO_OUTPUT_DIR)
    
    correct_grammar_in_subtitles(SUBTITLE_OUTPUT_FILE, INCORRECT_PARAGRAPH_FILE, CORRECTED_PARAGRAPH_FILE)
    
    with open(INCORRECT_PARAGRAPH_FILE, "r", encoding="utf-8") as f:
        incorrect_text = f.read()
    with open(CORRECTED_PARAGRAPH_FILE, "r", encoding="utf-8") as f:
        corrected_text = f.read()
    
    st.subheader("Incorrect Paragraph")
    st.write(incorrect_text)
    
    st.subheader("Corrected Paragraph")
    st.markdown(highlight_corrections(incorrect_text, corrected_text), unsafe_allow_html=True)
    
    st.success("Processing completed!")