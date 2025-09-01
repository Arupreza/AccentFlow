import os
import torch
import tempfile
import streamlit as st
from pydub import AudioSegment
from transformers import pipeline
from .ui import add_log

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