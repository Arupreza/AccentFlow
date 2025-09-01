import os
import difflib
import nltk
import streamlit as st
from .ui import add_log

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

def highlight_corrections(original: str, corrected: str) -> str:
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