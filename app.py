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
from text import *
from tts import *

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AccentFlow Demo App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .highlight {
        background-color: #f0f5ff;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .file-output {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        max-height: 200px;
        overflow-y: auto;
    }
    .success-message {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-message {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Create directories for storing files
@st.cache_resource
def initialize_directories():
    temp_dir = tempfile.mkdtemp()
    uploads_dir = os.path.join(temp_dir, "uploads")
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return temp_dir, uploads_dir, output_dir

temp_dir, uploads_dir, output_dir = initialize_directories()

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Upload"
if 'generated_files' not in st.session_state:
    st.session_state.generated_files = {}
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Function to add log messages
def add_log(message, type="info"):
    st.session_state.log_messages.append((message, type))
    if type == "error":
        st.error(message)

def process_video(video_path, output_dir, segment_duration, grammar_model, tts_model):
    """Main processing function."""
    try:
        if not os.path.exists(video_path):
            add_log(f"Video file not found: {video_path}", "error")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Reset generated files
        st.session_state.generated_files = {}
        
        # 1. Extract audio (20%)
        status_text.text("Extracting audio from video...")
        reference_path = extract_audio_segments(video_path, output_dir, segment_duration)
        if not reference_path:
            add_log("Failed to extract audio", "error")
            return False
        progress_bar.progress(20)
        st.session_state.progress = 20
        
        # 2. Transcribe video (50%)
        status_text.text("Transcribing video...")
        full_text, chunks = transcribe_video(video_path)
        if not chunks:
            add_log("Transcription failed or returned empty results", "error")
            return False
        progress_bar.progress(50)
        st.session_state.progress = 50
        
        # 3. Save subtitles (60%)
        status_text.text("Saving subtitles...")
        srt_file = os.path.join(output_dir, "subtitles.srt")
        srt_content = format_srt(chunks)
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_content)
        st.session_state.generated_files["subtitles"] = srt_file
        progress_bar.progress(60)
        st.session_state.progress = 60
        
        # 4. Correct grammar (80%)
        status_text.text("Correcting grammar...")
        incorrect_file = os.path.join(output_dir, "incorrect_text.txt")
        corrected_file = os.path.join(output_dir, "corrected_text.txt")
        success, incorrect_text, corrected_text = correct_grammar_in_subtitles(
            srt_file, incorrect_file, corrected_file, grammar_model
        )
        if not success:
            add_log("Grammar correction failed", "error")
            return False
        progress_bar.progress(80)
        st.session_state.progress = 80
        
        # 5. Generate TTS (100%)
        status_text.text("Generating speech from corrected text...")
        output_audio_path = os.path.join(output_dir, "output.wav")
        if not generate_tts(corrected_text, reference_path, output_audio_path, tts_model):
            add_log("TTS generation failed", "error")
            return False
        progress_bar.progress(100)
        st.session_state.progress = 100
        
        status_text.text("Processing complete!")
        st.session_state.processed = True
        
        # Store text for display
        st.session_state.incorrect_text = incorrect_text
        st.session_state.corrected_text = corrected_text
        st.session_state.highlighted_diff = highlight_corrections(incorrect_text, corrected_text)
        
        add_log("✅ All processing tasks completed successfully!", "success")
        return True
    except Exception as e:
        add_log(f"❌ Error processing video: {e}", "error")
        return False

# Function to get file download link
def get_file_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

# UI Layout
st.title("🎤 AccentFlow Demo App")
st.markdown("Upload a video to transcribe, correct grammar, and generate speech")

# Sidebar configuration
st.sidebar.header("Configuration")
hf_token_input = st.sidebar.text_input("Hugging Face Token (optional)", type="password", help="Enter your Hugging Face token for API access")
if hf_token_input:
    os.environ["HF_TOKEN"] = hf_token_input

segment_duration = st.sidebar.slider("Audio Segment Duration (seconds)", 5, 60, 10)

# Create tabs
tabs = st.tabs(["Upload", "Process", "Results", "Logs"])

# Upload Tab
with tabs[0]:
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        video_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.video_path = video_path
        st.success(f"Video uploaded successfully: {uploaded_file.name}")
        
        # Show video preview
        st.video(uploaded_file)
        
        # Add button to go to next tab
        if st.button("Continue to Processing"):
            st.session_state.current_tab = "Process"
            st.experimental_rerun()

# Process Tab
with tabs[1]:
    st.header("Process Video")
    
    if 'video_path' not in st.session_state:
        st.info("Please upload a video first")
    else:
        # Check if models are already loaded
        if 'grammar_model' not in st.session_state:
            st.session_state.grammar_model = load_grammar_model()
        
        if 'tts_model' not in st.session_state:
            st.session_state.tts_model = load_tts_model()
        
        # Display video info
        st.write(f"Video: {os.path.basename(st.session_state.video_path)}")
        st.write(f"Segment Duration: {segment_duration} seconds")
        
        # Process button
        if st.button("Start Processing"):
            with st.spinner("Processing video..."):
                success = process_video(
                    st.session_state.video_path, 
                    output_dir,
                    segment_duration,
                    st.session_state.grammar_model,
                    st.session_state.tts_model
                )
                
                if success:
                    st.success("Video processed successfully!")
                    st.session_state.current_tab = "Results"
                    st.experimental_rerun()
                else:
                    st.error("Error processing video. Check the logs tab for details.")

# Results Tab
with tabs[2]:
    st.header("Results")
    
    if not st.session_state.processed:
        st.info("No results available. Process a video first.")
    else:
        # Create columns for different outputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            st.markdown(f'<div class="file-output">{st.session_state.incorrect_text}</div>', unsafe_allow_html=True)
            
            if "subtitles" in st.session_state.generated_files:
                st.download_button(
                    "Download SRT Subtitles",
                    open(st.session_state.generated_files["subtitles"], "r", encoding="utf-8").read(),
                    file_name="subtitles.srt",
                    mime="text/plain"
                )
        
        with col2:
            st.subheader("Corrected Text")
            st.markdown(f'<div class="file-output">{st.session_state.highlighted_diff}</div>', unsafe_allow_html=True)
            
            if "corrected_text" in st.session_state.generated_files:
                st.download_button(
                    "Download Corrected Text",
                    open(st.session_state.generated_files["corrected_text"], "r", encoding="utf-8").read(),
                    file_name="corrected_text.txt",
                    mime="text/plain"
                )
        
        st.subheader("Generated Audio")
        if "tts_output" in st.session_state.generated_files:
            output_path = st.session_state.generated_files["tts_output"]
            st.audio(output_path)
            
            st.download_button(
                "Download Generated Audio",
                open(output_path, "rb").read(),
                file_name="generated_speech.wav",
                mime="audio/wav"
            )
        
        st.subheader("All Generated Files")
        for name, path in st.session_state.generated_files.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # Size in KB
                st.write(f"- {name}: {file_size:.1f} KB")

# Logs Tab
with tabs[3]:
    st.header("Processing Logs")
    
    if st.button("Clear Logs"):
        st.session_state.log_messages = []
        st.experimental_rerun()
    
    for msg, msg_type in st.session_state.log_messages:
        if msg_type == "error":
            st.error(msg)
        elif msg_type == "success":
            st.success(msg)
        elif msg_type == "warning":
            st.warning(msg)
        else:
            st.info(msg)

# Set active tab based on session state
active_tab = 0
if st.session_state.current_tab == "Process":
    active_tab = 1
elif st.session_state.current_tab == "Results":
    active_tab = 2
elif st.session_state.current_tab == "Logs":
    active_tab = 3

# Display info about the app
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses AI to process videos:
    - Extracts audio from videos
    - Transcribes speech using Whisper
    - Corrects grammar in transcriptions
    - Synthesizes speech from corrected text by voice cloning
    """
)

# Display technical info
st.sidebar.subheader("Technical Info")
st.sidebar.write(f"CPU/GPU: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    st.sidebar.write(f"GPU Model: {torch.cuda.get_device_name(0)}")
st.sidebar.write(f"Temp Directory: {temp_dir}")