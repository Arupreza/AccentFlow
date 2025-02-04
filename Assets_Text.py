import os
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer
from pydub import AudioSegment
import re
import difflib

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Define paths dynamically
OUTPUT_DIR = "saved_output"
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extracted_audio_segments")
FULL_AUDIO_OUTPUT = os.path.join(OUTPUT_DIR, "full_audio.wav")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

# Load Grammar Correction Model
model_name = "vennify/t5-base-grammar-correction"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
grammar_model = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)

def correct_text(text):
    """
    Use the grammar correction model directly.
    """
    return grammar_model(text)[0]['generated_text']

def remove_redundant_text(text):
    """
    Removes redundant occurrences of 'I :' in the corrected text.
    """
    return re.sub(r'\bI :\s*', '', text)

def transcribe_video(video_path):
    """
    Transcribe full text from video using OpenAI Whisper model with timestamps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-medium", 
        device=0 if device == "cuda" else -1,
        return_timestamps=True
    )
    result = asr_pipeline(video_path)
    return result["text"], result["chunks"]

def format_srt(transcription_chunks):
    """
    Format transcription into SRT format.
    """
    srt_content = []
    for idx, chunk in enumerate(transcription_chunks, start=1):
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
    
    return "\n".join(srt_content)

def extract_audio_segments(video_path, transcription_chunks, output_dir):
    """
    Extract audio segments corresponding to the subtitles.
    """
    full_audio = AudioSegment.from_file(video_path, format="mp4")
    
    for idx, chunk in enumerate(transcription_chunks, start=1):
        start_time = int(chunk["timestamp"][0] * 1000)
        end_time = int(chunk["timestamp"][1] * 1000)
        segment = full_audio[start_time:end_time]
        segment.export(os.path.join(output_dir, f"segment_{idx}.wav"), format="wav")
    
    full_audio.export(FULL_AUDIO_OUTPUT, format="wav")

def correct_grammar_in_subtitles(subtitle_file, incorrect_paragraph_file, corrected_paragraph_file):
    """
    Read subtitles, generate incorrect and corrected paragraphs.
    """
    with open(subtitle_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    incorrect_text = []
    corrected_text = []
    for line in lines:
        if "-->" not in line and line.strip().isdigit() is False:
            incorrect_text.append(line.strip())
            corrected_text_line = correct_text(line.strip())
            cleaned_text_line = remove_redundant_text(corrected_text_line)
            corrected_text.append(cleaned_text_line)
    
    with open(incorrect_paragraph_file, "w", encoding="utf-8") as file:
        file.write(" ".join(incorrect_text))
    
    with open(corrected_paragraph_file, "w", encoding="utf-8") as file:
        file.write(" ".join(corrected_text))

def highlight_corrections(original, corrected):
    """
    Highlight corrections in the corrected paragraph using HTML formatting.
    """
    diff = difflib.ndiff(original.split(), corrected.split())
    highlighted_text = " ".join(
        [f"<span style='color: red'>{word[2:]}</span>" if word.startswith("+ ") else word[2:]
         for word in diff if not word.startswith("- ")]
    )
    return highlighted_text
