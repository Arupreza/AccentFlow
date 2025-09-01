import torch
import streamlit as st

def inject_css():
    st.markdown("""
    <style>
        .main .block-container { padding-top: 2rem; }
        .stProgress > div > div > div > div { background-color: #4CAF50; }
        .highlight { background-color: #f0f5ff; padding: 20px; border-radius: 5px; margin: 10px 0; }
        .file-output { background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; max-height: 200px; overflow-y: auto; }
        .success-message { color: #4CAF50; font-weight: bold; }
        .error-message { color: #f44336; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def sidebar_config(temp_dir: str):
    st.sidebar.header("Configuration")
    hf_token_input = st.sidebar.text_input(
        "Hugging Face Token (optional)",
        type="password",
        help="Enter your Hugging Face token for API access"
    )
    segment_duration = st.sidebar.slider("Audio Segment Duration (seconds)", 5, 60, 10, key="segment_duration")

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

    return hf_token_input

def tabs_layout():
    return st.tabs(["Upload", "Process", "Results", "Logs"])

def set_active_tab():
    active_tab = {"Upload": 0, "Process": 1, "Results": 2, "Logs": 3}.get(st.session_state.current_tab, 0)
    # Streamlit currently doesn’t support programmatically switching tabs after render.
    # We keep this function for future Streamlit versions and semantic clarity.

def add_log(message: str, type: str = "info"):
    st.session_state.log_messages.append((message, type))
    if type == "error":
        st.error(message)