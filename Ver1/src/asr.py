# src/asr.py
from __future__ import annotations
import os
import tempfile
import traceback

import torch
import streamlit as st
from pydub import AudioSegment
from transformers import pipeline

from .ui import add_log


def _transcribe_file(wav_path: str, asr, generate_kwargs: dict):
    """
    Call HF ASR pipeline on a file path with proper kwargs.
    """
    return asr(wav_path, return_timestamps=True, generate_kwargs=generate_kwargs)


def transcribe_video_in_chunks(
    video_path: str,
    segment_duration: float = 10.0,
    force_lang: str | None = None,
):
    """
    Robust chunked transcription:
      - split with pydub
      - export each chunk to a temp WAV
      - run HF ASR pipeline on the WAV path
      - shift chunk timestamps by segment offset (absolute timeline)
    """
    try:
        if not os.path.isfile(video_path):
            add_log(f"ASR input video not found: {video_path}", "error")
            return "", []

        # Build segments
        audio = AudioSegment.from_file(video_path)
        seg_ms = max(1000, int(segment_duration * 1000))
        total_ms = len(audio)

        device = 0 if torch.cuda.is_available() else -1
        with st.spinner("Loading Whisper…"):
            asr = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-medium",  # upgradeable to large-v3 if you bump transformers
                device=device,
                return_timestamps=True,
            )

        # Hint Whisper about language if user forced en/ko/id
        generate_kwargs = {"task": "transcribe"}
        if force_lang in {"en", "ko", "id"}:
            generate_kwargs["language"] = force_lang

        texts, all_chunks = [], []

        # Iterate segments and keep absolute time
        for start_ms in range(0, total_ms, seg_ms):
            end_ms = min(start_ms + seg_ms, total_ms)
            seg = audio[start_ms:end_ms]

            # Write to a temporary WAV (most stable input for pipeline)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
                wav_path = t.name
            try:
                seg.export(wav_path, format="wav")
                with st.spinner(f"ASR {start_ms//seg_ms + 1}/{(total_ms + seg_ms - 1)//seg_ms}"):
                    result = _transcribe_file(wav_path, asr, generate_kwargs)
            finally:
                # Clean temp file
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            # Collect text
            texts.append(result.get("text", "") or "")

            # Collect and shift timestamps to absolute time
            chunks = result.get("chunks", []) or []
            if chunks:
                offset = start_ms / 1000.0
                for ch in chunks:
                    ts = ch.get("timestamp")
                    if (
                        isinstance(ts, (list, tuple))
                        and len(ts) == 2
                        and ts[0] is not None
                        and ts[1] is not None
                    ):
                        ch["timestamp"] = [float(ts[0]) + offset, float(ts[1]) + offset]
                        all_chunks.append(ch)

        full_text = " ".join(t for t in texts if t).strip()
        if not all_chunks:
            add_log("ASR returned no timestamped chunks; falling back to text only.", "warning")

        add_log("✅ Transcription complete", "success")
        return full_text, all_chunks

    except Exception as e:
        add_log(f"❌ ASR error: {e}", "error")
        add_log(traceback.format_exc(), "error")
        return "", []


def detect_language_from_text(text: str) -> str:
    """
    Light language detection using langdetect. Returns 'en','ko','id', etc. Falls back to 'en'.
    """
    try:
        if not text or not text.strip():
            return "en"
        from langdetect import detect
        lang = detect(text) or "en"
        return (lang.split("-")[0]).lower()
    except Exception:
        return "en"