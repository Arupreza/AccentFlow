import os
import ffmpeg
from pydub import AudioSegment
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


def extract_audio(video_path, output_audio_path="extracted_audio.wav"):
    """
    Extracts audio from a video file using FFmpeg.
    """
    ffmpeg.input(video_path).output(output_audio_path, format="wav").run(overwrite_output=True)
    return output_audio_path



def split_audio(audio_path, output_folder="audio_chunks", chunk_length_ms=30000):
    """
    Splits the extracted audio into 30-second chunks and saves them.
    """
    os.makedirs(output_folder, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i : i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_paths = []
    for idx, chunk in enumerate(chunks):
        chunk_filename = os.path.join(output_folder, f"chunk_{idx}.wav")
        chunk.export(chunk_filename, format="wav")
        chunk_paths.append(chunk_filename)
    
    return chunk_paths




def transcribe_audio(audio_chunks, output_folder="audio_chunks"):
    """
    Transcribes speech from multiple short audio chunks and saves text files.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0 if device == "cuda" else -1)

    transcriptions = []
    
    for idx, chunk in enumerate(audio_chunks):
        result = asr_pipeline(chunk)
        text = result["text"]

        # Save each transcription next to its corresponding audio file
        text_filename = os.path.join(output_folder, f"chunk_{idx}.txt")
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(text)

        transcriptions.append((chunk, text, text_filename))
    
    return transcriptions




def process_video(video_path, output_folder="output"):
    """
    Full pipeline: Extracts audio, splits it into chunks, and transcribes speech.
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing video: {video_path}")

    # Step 1: Extract audio
    audio_path = extract_audio(video_path, os.path.join(output_folder, "full_audio.wav"))
    print(f"Audio extracted: {audio_path}")

    # Step 2: Split into 30-second chunks
    audio_chunks = split_audio(audio_path, output_folder)
    print(f"Audio split into {len(audio_chunks)} chunks.")

    # Step 3: Transcribe each chunk and save next to audio
    transcript_info = transcribe_audio(audio_chunks, output_folder)

    print("\n✅ Process Completed!")
    for chunk, text, text_file in transcript_info:
        print(f"🔹 Audio: {chunk} -> 🔹 Text: {text_file}")

    return audio_chunks, transcript_info





# Example Usage
if __name__ == "__main__":
    video_file = "Sample_1.mp4"  # Replace with your video file path
    process_video(video_file, output_folder="transcriptions")





# Set Paths
data_dir = "/media/arupreza/Assets/LLM Projects/AccentFlow/transcriptions"
output_dir = os.path.join(data_dir, "corrected_texts")  # Save corrected texts here
os.makedirs(output_dir, exist_ok=True)



# Load Grammarly CoEdit-Large Model (T5-based)
model_name = "grammarly/coedit-large"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)




# Wrap Hugging Face model into LangChain
llm = HuggingFacePipeline.from_model_id(
    model_id=model_name, 
    task="text2text-generation",
    model_kwargs={"trust_remote_code": True}
)



# Define LangChain Prompt
grammar_prompt = PromptTemplate(
    input_variables=["text"],
    template="Fix grammar: {text}"
)



# LangChain Chain for Grammar Correction
grammar_chain = LLMChain(llm=llm, prompt=grammar_prompt)




def correct_grammar_langchain(text):
    """
    Uses LangChain with Grammarly CoEdit-Large to correct grammar.
    """
    response = grammar_chain.run({"text": text})
    return response.strip()



def process_text_files():
    """
    Processes all .txt files in the input directory using LangChain, corrects grammar, and saves output.
    """
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".txt", "_corrected.txt"))

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            corrected_text = correct_grammar_langchain(text)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(corrected_text)

            print(f"✅ Processed: {filename} → {output_path}")