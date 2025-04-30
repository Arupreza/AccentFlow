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
import nltk
import numpy as np
from dotenv import load_dotenv

# Download NLTK data
nltk.download('punkt')

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Audio Processing App",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
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

# Initialize directories
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

# Load TTS Model
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

# Core functions
def extract_audio_segments(video_path: str, output_dir: str, segment_duration: float):
    """Extract audio from video and split into segments."""
    try:
        if not os.path.exists(video_path):
            add_log(f"Video file not found: {video_path}", "error")
            return None
            
        os.makedirs(output_dir, exist_ok=True)
        full_audio = AudioSegment.from_file(video_path)
        
        full_wav_path = os.path.join(output_dir, "full_audio.wav")
        full_audio.export(full_wav_path, format="wav")
        st.session_state.generated_files["full_audio"] = full_wav_path

        segment_duration_ms = segment_duration * 1000
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

def transcribe_segment(audio_segment: AudioSegment, asr_pipeline):
    """Transcribe a single audio segment."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_segment.export(tmp.name, format="wav")
        tmp_path = tmp.name
    try:
        result = asr_pipeline(tmp_path, return_timestamps=True)
        return result.get("text", ""), result.get("chunks", [])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def transcribe_video_in_chunks(video_path: str, segment_duration: float = 10.0):
    """Transcribe video by breaking it into audio segments."""
    try:
        segments = []
        full_audio = AudioSegment.from_file(video_path)
        seg_len_ms = int(segment_duration * 1000)
        for start in range(0, len(full_audio), seg_len_ms):
            end = start + seg_len_ms
            segments.append(full_audio[start:end])

        device_id = 0 if torch.cuda.is_available() else -1
        with st.spinner("Loading Whisper ASR model..."):
            asr = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-medium",
                device=device_id,
                return_timestamps=True
            )

        all_texts = []
        all_chunks = []
        for idx, seg in enumerate(segments, start=1):
            with st.spinner(f"Transcribing segment {idx}/{len(segments)}..."):
                text, chunks = transcribe_segment(seg, asr)
                all_texts.append(text)
                all_chunks.extend(chunks)

        full_text = " ".join(t for t in all_texts if t)
        add_log("✅ Transcription complete", "success")
        return full_text, all_chunks
    except Exception as e:
        add_log(f"❌ Error transcribing video: {e}", "error")
        return "", []

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

def chunk_text(text: str, max_sentences: int = 3, overlap: int = 0) -> list[str]:
    """Split text into chunks of up to max_sentences sentences."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    start = 0
    total = len(sentences)
    while start < total:
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        if end >= total:
            break
        start += max_sentences - overlap
    return chunks

def correct_chunk(model, chunk: str, max_length: int = 512) -> str:
    """Correct grammar in a text chunk."""
    try:
        if not chunk.strip():
            return chunk
        out = model(chunk, max_length=max_length, truncation=True)
        return out[0]["generated_text"]
    except Exception as e:
        add_log(f"Error correcting chunk: {e}", "error")
        return chunk

def correct_grammar_in_subtitles(subtitle_file, incorrect_paragraph_file, corrected_paragraph_file, grammar_model):
    """Correct grammar in subtitles and save results."""
    try:
        if not os.path.exists(subtitle_file):
            add_log(f"Subtitle file not found: {subtitle_file}", "error")
            return False, "", ""

        with open(subtitle_file, "r", encoding="utf-8") as file:
            lines = file.readlines()

        incorrect_text = []
        for line in lines:
            if "-->" not in line and not line.strip().isdigit() and line.strip():
                incorrect_text.append(line.strip())

        incorrect_para = " ".join(incorrect_text)
        chunks = chunk_text(incorrect_para, max_sentences=3, overlap=0)
        corrected_chunks = []
        for i, chunk in enumerate(chunks, 1):
            with st.spinner(f"Correcting chunk {i}/{len(chunks)}..."):
                corrected_chunks.append(correct_chunk(grammar_model, chunk))

        corrected_para = " ".join(corrected_chunks)

        os.makedirs(os.path.dirname(incorrect_paragraph_file), exist_ok=True)
        os.makedirs(os.path.dirname(corrected_paragraph_file), exist_ok=True)

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

def trim_reference(input_path, output_path, seconds=5):
    """Trim reference audio to specified length."""
    try:
        if not os.path.exists(input_path):
            add_log(f"Reference audio file not found: {input_path}", "error")
            return False
        audio = AudioSegment.from_file(input_path)
        if len(audio) < seconds * 1000:
            add_log(f"Warning: Audio file is shorter than {seconds} seconds.", "warning")
            trimmed_audio = audio
        else:
            trimmed_audio = audio[:seconds * 1000]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trimmed_audio.export(output_path, format="wav")
        st.session_state.generated_files["trimmed_ref"] = output_path
        add_log("✅ Reference audio trimmed successfully", "success")
        return True
    except Exception as e:
        add_log(f"❌ Error trimming reference audio: {e}", "error")
        return False

def generate_tts(text, reference_path, output_path, tts_model):
    """Generate speech from text using TTS model."""
    try:
        if not os.path.exists(reference_path):
            add_log(f"Reference audio not found: {reference_path}", "error")
            return False

        trimmed_ref_path = os.path.join(os.path.dirname(reference_path), "trimmed_ref.wav")
        os.makedirs(os.path.dirname(trimmed_ref_path), exist_ok=True)
        if not trim_reference(reference_path, trimmed_ref_path, seconds=5):
            return False

        if tts_model is None:
            add_log("TTS model not loaded", "error")
            return False

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

def highlight_corrections(original, corrected):
    """Highlight differences between original and corrected text."""
    try:
        diff = difflib.ndiff(original.split(), corrected.split())
        highlighted_text = " ".join(
            [f"<span style='color: red; font-weight: bold;'>{word[2:]}</span>" if word.startswith("+ ") else word[2:]
            for word in diff if not word.startswith("- ")]
        )
        return highlighted_text
    except Exception as e:
        add_log(f"Error highlighting corrections: {e}", "error")
        return corrected

def process_video(video_path, output_dir, segment_duration, grammar_model, tts_model):
    """Main processing function."""
    try:
        if not os.path.exists(video_path):
            add_log(f"Video file not found: {video_path}", "error")
            return False

        os.makedirs(output_dir, exist_ok=True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        st.session_state.generated_files = {}

        # Extract audio
        status_text.text("Extracting audio from video...")
        reference_path = extract_audio_segments(video_path, output_dir, segment_duration)
        if not reference_path:
            return False
        progress_bar.progress(20)
        st.session_state.progress = 20

        # Transcribe video
        status_text.text("Transcribing video...")
        full_text, chunks = transcribe_video_in_chunks(video_path, segment_duration)
        if not chunks:
            return False
        progress_bar.progress(50)
        st.session_state.progress = 50

        # Save subtitles
        status_text.text("Saving subtitles...")
        srt_file = os.path.join(output_dir, "subtitles.srt")
        srt_content = format_srt(chunks)
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_content)
        st.session_state.generated_files["subtitles"] = srt_file
        progress_bar.progress(60)
        st.session_state.progress = 60

        # Correct grammar
        status_text.text("Correcting grammar...")
        incorrect_file = os.path.join(output_dir, "incorrect_text.txt")
        corrected_file = os.path.join(output_dir, "corrected_text.txt")
        success, incorrect_text, corrected_text = correct_grammar_in_subtitles(
            srt_file, incorrect_file, corrected_file, grammar_model
        )
        if not success:
            return False
        progress_bar.progress(80)
        st.session_state.progress = 80

        # Generate TTS
        status_text.text("Generating speech from corrected text...")
        output_audio_path = os.path.join(output_dir, "output.wav")
        if not generate_tts(corrected_text, reference_path, output_audio_path, tts_model):
            return False
        progress_bar.progress(100)
        st.session_state.progress = 100

        status_text.text("Processing complete!")
        st.session_state.processed = True
        st.session_state.incorrect_text = incorrect_text
        st.session_state.corrected_text = corrected_text
        st.session_state.highlighted_diff = highlight_corrections(incorrect_text, corrected_text)
        add_log("✅ All processing tasks completed successfully!", "success")
        return True
    except Exception as e:
        add_log(f"❌ Error processing video: {e}", "error")
        return False

def get_file_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'

# UI Layout
st.title("🎤 Audio Processing App")
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
        video_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.video_path = video_path
        st.success(f"Video uploaded successfully: {uploaded_file.name}")
        st.video(uploaded_file)
        
        if st.button("Continue to Processing"):
            st.session_state.current_tab = "Process"
            st.experimental_rerun()

# Process Tab
with tabs[1]:
    st.header("Process Video")
    
    if 'video_path' not in st.session_state:
        st.info("Please upload a video first")
    else:
        if 'grammar_model' not in st.session_state:
            st.session_state.grammar_model = load_grammar_model()
        
        if 'tts_model' not in st.session_state:
            st.session_state.tts_model = load_tts_model()
        
        st.write(f"Video: {os.path.basename(st.session_state.video_path)}")
        st.write(f"Segment Duration: {segment_duration} seconds")
        
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
                file_size = os.path.getsize(path) / 1024
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

# Set active tab
active_tab = {"Upload": 0, "Process": 1, "Results": 2, "Logs": 3}.get(st.session_state.current_tab, 0)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    """
    This app uses AI to process videos:
    - Extracts audio from videos
    - Transcribes speech using Whisper
    - Corrects grammar in transcriptions
    - Synthesizes speech from corrected text
    """
)

st.sidebar.subheader("Technical Info")
st.sidebar.write(f"CPU/GPU: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    st.sidebar.write(f"GPU Model: {torch.cuda.get_device_name(0)}")
st.sidebar.write(f"Temp Directory: {temp_dir}")