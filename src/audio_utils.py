import os
import streamlit as st
from pydub import AudioSegment
from .ui import add_log

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

        segment_duration_ms = int(segment_duration * 1000)
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