import os
import streamlit as st

from .ui import add_log
from .audio_utils import extract_audio_segments, trim_reference
from .asr import transcribe_video_in_chunks
from .text_utils import format_srt, correct_grammar_in_subtitles, highlight_corrections

def generate_tts(text: str, reference_path: str, output_path: str, tts_model):
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
            # Note: XTTS expects a *list* for multi-speaker; single still works as [path]
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

def process_video(video_path, output_dir, segment_duration, grammar_model, tts_model):
    """Main processing function gluing all steps together."""
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

        # Transcribe
        status_text.text("Transcribing video...")
        full_text, chunks = transcribe_video_in_chunks(video_path, segment_duration)
        if not chunks:
            return False
        progress_bar.progress(50)
        st.session_state.progress = 50

        # Save SRT
        status_text.text("Saving subtitles...")
        srt_file = os.path.join(output_dir, "subtitles.srt")
        srt_content = format_srt(chunks)
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_content)
        st.session_state.generated_files["subtitles"] = srt_file
        progress_bar.progress(60)
        st.session_state.progress = 60

        # Grammar correction
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

        # TTS
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