import os
import tempfile
import nltk
import streamlit as st

from Ver1.src.config import SEGMENT_DURATION, SUPPORTED_LANGS
from Ver1.src.ui import inject_css, sidebar, tabs_layout, add_log
from Ver1.src.models import load_grammar_model, load_tts_model
from Ver1.src.pipeline import process_video

# one-time setup
nltk.download("punkt", quiet=True)
st.set_page_config(page_title="Audio Processing App", page_icon="🎤", layout="wide")
inject_css()

# session state
st.session_state.setdefault("processed", False)
st.session_state.setdefault("current_tab", "Upload")
st.session_state.setdefault("generated_files", {})
st.session_state.setdefault("log_messages", [])
st.session_state.setdefault("progress", 0)

@st.cache_resource
def _dirs():
    tmp = tempfile.mkdtemp()
    up = os.path.join(tmp, "uploads")
    out = os.path.join(tmp, "output")
    os.makedirs(up, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    return tmp, up, out

temp_dir, uploads_dir, output_dir = _dirs()
# set debug dir so logs persist to file
st.session_state["debug_dir"] = output_dir

sidebar(temp_dir)

st.title("🎤 Audio Processing App")
st.write("Upload → ASR → (optional **translate**) → (English **grammar**) → **aligned TTS** → **replace audio**. No lip-sync.")

tabs = tabs_layout()

# Upload
with tabs[0]:
    st.header("Upload Video")
    f = st.file_uploader("Choose a video", type=["mp4", "mov", "mkv", "avi"])
    if f is not None:
        vpath = os.path.join(uploads_dir, f.name)
        with open(vpath, "wb") as w: w.write(f.getbuffer())
        st.session_state.video_path = vpath
        st.success(f"Uploaded: {f.name}")
        st.video(vpath)
        if st.button("Continue to Processing"):
            st.session_state.current_tab = "Process"
            st.experimental_rerun()

# Process
with tabs[1]:
    st.header("Process Video")
    if "video_path" not in st.session_state:
        st.info("Please upload a video first.")
    else:
        if "grammar_model" not in st.session_state:
            st.session_state.grammar_model = load_grammar_model()
        if "tts_model" not in st.session_state:
            st.session_state.tts_model = load_tts_model()

        st.write(f"Video: **{os.path.basename(st.session_state.video_path)}**")
        st.write(f"ASR segment: **{SEGMENT_DURATION}s**")

        col1, col2 = st.columns(2)
        with col1:
            src_choice = st.selectbox(
                "Source language (for ASR)",
                options=[k for k, _ in SUPPORTED_LANGS],
                format_func=lambda k: dict(SUPPORTED_LANGS)[k]
            )
        with col2:
            tgt_choice = st.selectbox(
                "Target language for TTS",
                options=["same", "en", "ko", "id"],
                format_func=lambda k: {"same":"Same as source","en":"English","ko":"Korean","id":"Indonesian"}[k]
            )

        translate_flag = st.checkbox("Translate transcript to the target language before TTS", value=True)

        if st.button("Start Processing"):
            with st.spinner("Processing..."):
                ok = process_video(
                    video_path=st.session_state.video_path,
                    output_dir=output_dir,
                    segment_duration=SEGMENT_DURATION,
                    grammar_model=st.session_state.grammar_model,
                    tts_model=st.session_state.tts_model,
                    source_lang_override=None if src_choice == "auto" else src_choice,
                    target_lang_choice=tgt_choice,
                    translate_before_tts=translate_flag,
                )
                if ok:
                    st.success("Done!")
                    st.session_state.current_tab = "Results"
                    st.experimental_rerun()
                else:
                    st.error("Error. See Logs tab.")

# Results
with tabs[2]:
    st.header("Results")
    if not st.session_state.get("processed"):
        st.info("Process a video first.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original Transcript")
            st.markdown(
                f"<div class='file-output'>{st.session_state.get('incorrect_text','')}</div>",
                unsafe_allow_html=True
            )
            srt = st.session_state.generated_files.get("subtitles")
            if srt and os.path.exists(srt):
                st.download_button("Download SRT", open(srt, "r", encoding="utf-8").read(),
                                   file_name="subtitles.srt", mime="text/plain")
        with c2:
            st.subheader("Final Text (after GEC/translation)")
            st.markdown(
                f"<div class='file-output'>{st.session_state.get('highlighted_diff','')}</div>",
                unsafe_allow_html=True
            )
            corr = st.session_state.generated_files.get("corrected_text")
            if corr and os.path.exists(corr):
                st.download_button("Download Final Text",
                                   open(corr, "r", encoding="utf-8").read(),
                                   file_name="final_text.txt", mime="text/plain")

        st.subheader("Aligned TTS Audio")
        tts = st.session_state.generated_files.get("tts_output")
        if tts and os.path.exists(tts):
            st.audio(tts)
            st.download_button("Download Audio", open(tts, "rb").read(),
                               file_name="generated_speech.wav", mime="audio/wav")

        st.subheader("Merged Video (original video + aligned audio)")
        mv = st.session_state.generated_files.get("merged_video")
        if mv and os.path.exists(mv):
            st.video(mv)
            with open(mv, "rb") as f:
                st.download_button("Download Video", f, file_name="merged_video.mp4", mime="video/mp4")
        else:
            st.warning("Merged video not found.")

        st.subheader("All Generated Files")
        for k, p in st.session_state.generated_files.items():
            if os.path.exists(p):
                st.write(f"- {k}: {os.path.getsize(p)/1024:.1f} KB — {p}")

# Logs
with tabs[3]:
    st.header("Logs")
    if st.button("Clear Logs"):
        st.session_state.log_messages = []
        st.experimental_rerun()

    # Show in-app log messages
    for msg, kind in st.session_state.log_messages:
        getattr(st, {"error":"error","success":"success","warning":"warning"}.get(kind,"info"))(msg)

    # Quick diagnostics to surface environment issues
    with st.expander("Diagnostics"):
        if st.button("Run quick diagnostics"):
            import subprocess, shutil, torch, transformers, TTS, sentencepiece, sacremoses
            from langdetect import detect
            from Ver1.src.audio_utils import _which_ffmpeg
            st.write("Python:", os.sys.version)
            st.write("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
            if torch.cuda.is_available():
                st.write("GPU:", torch.cuda.get_device_name(0))
            st.write("transformers:", transformers.__version__)
            st.write("TTS:", TTS.__version__)
            st.write("sentencepiece:", sentencepiece.__version__)
            st.write("sacremoses:", sacremoses.__version__)
            st.write("langdetect OK →", detect("Hello world"))
            st.write("ffmpeg in PATH:", _which_ffmpeg() or "NOT FOUND")
            if _which_ffmpeg():
                try:
                    out = subprocess.run(["ffmpeg","-version"], capture_output=True, text=True)
                    st.code(out.stdout.splitlines()[0] if out.stdout else out.stderr)
                except Exception as e:
                    st.write("ffmpeg check failed:", e)