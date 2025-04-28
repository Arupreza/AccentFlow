import os
import streamlit as st
from dotenv import load_dotenv
import torch
from pydub import AudioSegment
from TTS.api import TTS

# Custom utilities
from Assets_Text import (
    transcribe_video,
    format_srt,
    extract_audio_segments,
    correct_grammar_in_subtitles,
    highlight_corrections,
)

# Load environment variables
load_dotenv()

# --------------------
# PyTorch safe-unpickle allowlist (optional)
# --------------------
import TTS.tts.configs.xtts_config as _xtts_cfg
import TTS.tts.models.xtts as _xtts_models
import TTS.config.shared_configs as _shared_cfg

_add_sg = getattr(torch.serialization, "add_safe_globals", None)
if callable(_add_sg):
    _add_sg([
        _xtts_cfg.XttsConfig,
        _xtts_models.XttsAudioConfig,
        _shared_cfg.BaseDatasetConfig,
        _xtts_models.XttsArgs,
    ])
else:
    st.warning("⚠️ add_safe_globals not available; skipping safe-unpickle registration.")

# --------------------
# Paths
# --------------------
OUTPUT_DIR        = "saved_output"
AUDIO_SEG_DIR     = os.path.join(OUTPUT_DIR, "extracted_audio_segments")
SUBTITLE_FILE     = os.path.join(OUTPUT_DIR, "transcribed_subtitles.srt")
INCORRECT_FILE    = os.path.join(OUTPUT_DIR, "incorrect_paragraph.txt")
CORRECTED_FILE    = os.path.join(OUTPUT_DIR, "corrected_paragraph.txt")

# Adjust this to where your full_audio.wav lives
AUDIO_DIR   = "/media/arupreza/Assets/AccentFlow_App_0.0/AccentFlow/saved_output"
FULL_WAV    = os.path.join(AUDIO_DIR, "full_audio.wav")
TRIM_WAV    = os.path.join(AUDIO_DIR, "trimmed_ref.wav")
OUTPUT_WAV  = os.path.join(AUDIO_DIR, "output.wav")

# Ensure directories exist
for d in (OUTPUT_DIR, AUDIO_SEG_DIR, AUDIO_DIR):
    os.makedirs(d, exist_ok=True)

# --------------------
# Helpers
# --------------------
def trim_reference(input_wav: str, output_wav: str, seconds: int = 5):
    """Trim the first `seconds` seconds from input_wav → output_wav."""
    audio = AudioSegment.from_wav(input_wav)
    snippet = audio[: seconds * 1000]
    snippet.export(output_wav, format="wav")

def generate_tts(text: str):
    """Inline XTTS-v2 call—no subprocess."""
    if not os.path.exists(FULL_WAV):
        st.error(f"❌ full_audio.wav not found at: {FULL_WAV}")
        return

    # 1) Trim 5 seconds of reference
    trim_reference(FULL_WAV, TRIM_WAV, seconds=5)

    # 2) Load model
    use_gpu = torch.cuda.is_available()
    st.info(f"Loading XTTS-v2 on {'GPU' if use_gpu else 'CPU'}…")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

    # 3) Synthesize
    st.info("Synthesizing… this may take a moment.")
    tts.tts_to_file(
        text=text,
        speaker_wav=[TRIM_WAV],
        language="en",
        file_path=OUTPUT_WAV
    )
    st.success("✅ TTS synthesis complete")

# --------------------
# Streamlit App
# --------------------
st.title("Grammar Correction & XTTS-v2 Demo")

video = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
if video:
    # Save uploaded video
    vid_path = os.path.join(OUTPUT_DIR, video.name)
    with open(vid_path, "wb") as f:
        f.write(video.read())
    st.success("✔️ Video uploaded")

    # Transcribe video and extract audio segments
    text, chunks = transcribe_video(vid_path)
    srt = format_srt(chunks)
    with open(SUBTITLE_FILE, "w", encoding="utf-8") as f:
        f.write(srt)
    extract_audio_segments(vid_path, chunks, AUDIO_SEG_DIR)

    # Perform grammar correction
    correct_grammar_in_subtitles(SUBTITLE_FILE, INCORRECT_FILE, CORRECTED_FILE)
    original  = open(INCORRECT_FILE, encoding="utf-8").read()
    corrected = open(CORRECTED_FILE, encoding="utf-8").read()

    st.subheader("Original Text")
    st.write(original)
    st.subheader("Corrected Text")
    st.markdown(highlight_corrections(original, corrected), unsafe_allow_html=True)
    st.success("✔️ Grammar correction done")

    # TTS synthesis
    st.subheader("Generate Voice")
    if st.button("Speak corrected text"):
        generate_tts(corrected)
        if os.path.exists(OUTPUT_WAV):
            st.audio(OUTPUT_WAV)
