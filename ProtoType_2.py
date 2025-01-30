import os
import streamlit as st
import ffmpeg
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence

def setup_page():
    st.set_page_config(page_title="AccentFlow - AI Speech Correction", layout="wide")
    st.title("🎙️ AccentFlow - AI-Powered Speech Correction")
    st.write("Upload a video, and AccentFlow will transcribe and correct your speech automatically.")

# Function to extract audio from video
def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    ffmpeg.input(video_path).output(output_audio_path, format="wav").run(overwrite_output=True)
    return output_audio_path

# Function to split audio into 30-second chunks
def split_audio(audio_path, output_dir="Streamlit_out/old", chunk_length_ms=30000):
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    
    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_dir, f"chunk_{idx}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_paths.append(chunk_filename)
    
    return chunk_paths

# Function to transcribe audio
def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if device == "cuda" else -1)
    result = asr_pipeline(audio_path)
    return result["text"]

# Load Grammar Correction Model
model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text2text-generation", model_kwargs={"trust_remote_code": True})

grammar_prompt = PromptTemplate(input_variables=["text"], template="Fix grammar: {text}")
grammar_chain = LLMChain(llm=llm, prompt=grammar_prompt)

# Function to correct grammar
def correct_grammar(text):
    response = grammar_chain.run({"text": text})
    return response.strip()

def process_audio_chunks(chunk_paths, old_dir="Streamlit_out/old", new_dir="Streamlit_out/new"):
    os.makedirs(new_dir, exist_ok=True)
    for chunk_path in chunk_paths:
        text = transcribe_audio(chunk_path)
        text_path = chunk_path.replace(".wav", ".txt")
        with open(os.path.join(old_dir, os.path.basename(text_path)), "w") as f:
            f.write(text)
        corrected_text = correct_grammar(text)
        new_text_path = os.path.join(new_dir, os.path.basename(text_path))
        with open(new_text_path, "w") as f:
            f.write(corrected_text)

def concatenate_texts(directory, output_file):
    with open(output_file, "w") as outfile:
        texts = []
        for file in sorted(os.listdir(directory)):
            if file.endswith(".txt"):
                with open(os.path.join(directory, file), "r") as infile:
                    texts.append(infile.read().strip())
        outfile.write(" ".join(texts))

def main():
    setup_page()
    uploaded_video = st.file_uploader("📂 Upload your video file", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.video(uploaded_video)
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.write("🔊 Extracting Audio...")
        audio_path = extract_audio(video_path)
        
        st.write("🔊 Splitting Audio into Chunks...")
        chunk_paths = split_audio(audio_path)
        
        st.write("📝 Processing Audio Chunks...")
        process_audio_chunks(chunk_paths)
        
        old_texts_path = "Streamlit_out/old/all_old_transcription.txt"
        new_texts_path = "Streamlit_out/new/all_corrected_transcription.txt"
        concatenate_texts("Streamlit_out/old", old_texts_path)
        concatenate_texts("Streamlit_out/new", new_texts_path)
        
        with open(old_texts_path, "r") as f:
            old_text = f.read()
        with open(new_texts_path, "r") as f:
            new_text = f.read()
        
        st.markdown("### 🔤 Original Transcription:")
        st.text_area("", old_text, height=300)
        
        st.markdown("### ✅ Corrected Text:")
        st.text_area("", new_text, height=300)
        
        st.download_button("📥 Download Corrected Text", new_text, file_name="corrected_transcription.txt", mime="text/plain")
        
if __name__ == "__main__":
    main()
