import os
import torch
import streamlit as st
import nltk
from dotenv import load_dotenv

from src.config import init_dirs, init_session_state, set_page_config
from src.ui import inject_css, sidebar_config, tabs_layout, set_active_tab
from src.models import load_grammar_model, load_tts_model
from src.pipeline import process_video
from src.ui import add_log

# ---------- one-time setup ----------
nltk.download('punkt')
load_dotenv()
set_page_config()
inject_css()

# ---------- dirs & session ----------
temp_dir, uploads_dir, output_dir = init_dirs()
init_session_state()

# ---------- app title ----------
st.title("🎤 Audio Processing App")
st.markdown("Upload a video to transcribe, correct grammar, and generate speech")

# ---------- sidebar ----------
hf_token = sidebar_config(temp_dir)
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ---------- tabs ----------
tabs = tabs_layout()

# ---- Upload Tab ----
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

# ---- Process Tab ----
with tabs[1]:
    st.header("Process Video")

    if 'video_path' not in st.session_state:
        st.info("Please upload a video first")
    else:
        if 'grammar_model' not in st.session_state:
            st.session_state.grammar_model = load_grammar_model()

        if 'tts_model' not in st.session_state:
            st.session_state.tts_model = load_tts_model()

        segment_duration = st.session_state.get("segment_duration", 10)
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
                    st.error("Error processing video. Check the Logs tab for details.")

# ---- Results Tab ----
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

# ---- Logs Tab ----
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

# set active tab
set_active_tab()