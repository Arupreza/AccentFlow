# src/ui.py
import os
import datetime
import torch
import streamlit as st

def add_log(message: str, level: str = "info"):
    """
    Append to Streamlit logs AND persist to a debug file if st.session_state['debug_dir'] is set.
    """
    st.session_state.setdefault("log_messages", [])
    st.session_state["log_messages"].append((message, level))

    # Persist to file for easier debugging
    try:
        out_dir = st.session_state.get("debug_dir")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "debug.log"), "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().isoformat(timespec="seconds")
                f.write(f"[{ts}] [{level.upper()}] {message}\n")
    except Exception:
        # never break UI logging on file issues
        pass

def inject_css():
    st.markdown("""
    <style>
      .main .block-container { padding-top: 2rem; }
      .stProgress > div > div > div > div { background-color: #4CAF50; }
      .file-output {
        background:#f8f9fa; padding:10px; border-radius:6px; max-height:250px; overflow:auto;
      }
    </style>
    """, unsafe_allow_html=True)

def sidebar(temp_dir: str):
    st.sidebar.header("About")
    st.sidebar.info(
        "- Extract audio\n"
        "- Transcribe (Whisper)\n"
        "- Optional **translate** (EN↔KO / EN↔ID; KO↔ID via EN)\n"
        "- English grammar correction\n"
        "- Timestamp-aligned TTS (XTTS, voice clone)\n"
        "- Replace video audio (FFmpeg)\n"
        "**No lip-sync model**"
    )
    st.sidebar.subheader("Device")
    st.sidebar.write("GPU" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        try:
            st.sidebar.write(torch.cuda.get_device_name(0))
        except Exception:
            pass
    st.sidebar.subheader("Temp")
    st.sidebar.write(temp_dir)

def tabs_layout():
    return st.tabs(["Upload", "Process", "Results", "Logs"])