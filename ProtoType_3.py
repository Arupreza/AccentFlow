import os
import streamlit as st
import ffmpeg
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from pydub import AudioSegment
import difflib

# Define save directories
MODEL_DIR = "Streamlit_out"
OLD_DIR = os.path.join(MODEL_DIR, "old")
NEW_DIR = os.path.join(MODEL_DIR, "new")
MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pth")

# Ensure directories exist
os.makedirs(OLD_DIR, exist_ok=True)
os.makedirs(NEW_DIR, exist_ok=True)

def setup_page():
    st.set_page_config(page_title="AccentFlow - AI Speech Correction", layout="wide")
    st.title("🎙️ AccentFlow - AI-Powered Speech Correction")
    st.write("Upload a video, and AccentFlow will transcribe and correct your speech automatically.")

# Extract audio from video
def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    ffmpeg.input(video_path).output(output_audio_path, format="wav").run(overwrite_output=True)
    return output_audio_path

# Split audio into chunks
def split_audio(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_filename = os.path.join(OLD_DIR, f"chunk_{idx}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_paths.append(chunk_filename)

    return chunk_paths

# Transcribe audio
def transcribe_audio(audio_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if device == "cuda" else -1)
    result = asr_pipeline(audio_path)
    return result["text"]

# Load Grammar Correction Model
model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load existing model if available
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text2text-generation", model_kwargs={"trust_remote_code": True})

# Default prompt
default_prompt = "Fix the grammar while maintaining the sentence structure and meaning."

# Grammar correction function
def correct_grammar(text, user_prompt=None):
    prompt = user_prompt if user_prompt else default_prompt
    grammar_prompt = PromptTemplate(input_variables=["text"], template=f"{prompt}\nText: {{text}}")
    grammar_chain = LLMChain(llm=llm, prompt=grammar_prompt)
    return grammar_chain.run({"text": text}).strip()

# Check similarity between original and corrected text
def check_similarity(original, corrected):
    similarity_ratio = difflib.SequenceMatcher(None, original, corrected).ratio()
    return similarity_ratio

# Process audio chunks
def process_audio_chunks(chunk_paths):
    for chunk_path in chunk_paths:
        text = transcribe_audio(chunk_path)
        text_filename = os.path.basename(chunk_path).replace(".wav", ".txt")

        old_text_path = os.path.join(OLD_DIR, text_filename)
        with open(old_text_path, "w") as f:
            f.write(text)

        corrected_text = correct_grammar(text)
        similarity = check_similarity(text, corrected_text)

        # Ensure corrected text is similar enough before saving
        if similarity < 0.75:
            corrected_text = correct_grammar(text)  # Retry correction

        new_text_path = os.path.join(NEW_DIR, text_filename)
        with open(new_text_path, "w") as f:
            f.write(corrected_text)

# Concatenate all transcriptions
def concatenate_texts(directory):
    texts = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r") as infile:
                texts.append(infile.read().strip())
    return " ".join(texts)

# Retrain the model
def retrain_model(original_text, corrected_text):
    input_ids = tokenizer(original_text, return_tensors="pt").input_ids
    target_ids = tokenizer(corrected_text, return_tensors="pt").input_ids

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=target_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    torch.save(model, MODEL_PATH)

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

        st.write("🔊 Splitting Audio into 30-Second Chunks...")
        chunk_paths = split_audio(audio_path)

        st.write("📝 Processing Audio Chunks...")
        process_audio_chunks(chunk_paths)

        old_text = concatenate_texts(OLD_DIR)
        new_text = concatenate_texts(NEW_DIR)

        # User-specified prompt
        user_prompt = st.text_area("✍️ Customize the correction prompt (optional)", "", height=100)

        # Generate correction with user prompt
        if user_prompt:
            st.write("🔍 Applying your custom prompt for grammar correction...")
            new_text = correct_grammar(old_text, user_prompt)

        st.markdown("### ✅ Final Corrected Text:")
        st.text_area("", new_text, height=300)

        # Take User Feedback
        feedback = st.radio("How was the correction?", ("Best", "Good", "Bad"))

        # If feedback is "Bad", try again
        if feedback == "Bad":
            st.write("🔄 Retrying correction...")
            corrected_text_retry = correct_grammar(old_text, user_prompt)
            similarity = check_similarity(old_text, corrected_text_retry)

            # If retry similarity is still low, keep the original correction
            if similarity < 0.75:
                corrected_text_retry = new_text

            st.markdown("### 🔄 Updated Corrected Text (After Retrying):")
            st.text_area("", corrected_text_retry, height=300)

            # Retrain model with the retried version
            retrain_model(old_text, corrected_text_retry)
            st.success("✅ Model updated and saved in Streamlit_out/saved_model.pth")

        st.download_button("📥 Download Corrected Text", new_text, file_name="corrected_transcription.txt", mime="text/plain")

if __name__ == "__main__":
    main()
