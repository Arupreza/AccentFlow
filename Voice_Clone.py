import os
import torch
import pysrt
import pandas as pd
from TTS.api import TTS

# Ensure GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define dataset path using saved_output/extracted_audio_segments
dataset_path = "saved_output/extracted_audio_segments"  # Adjust to actual dataset directory
srt_file_path = os.path.join(dataset_path, "transcribed_subtitles.srt")  # Adjust if filename differs
metadata_path = os.path.join(dataset_path, "metadata.csv")
config_path = os.path.join(dataset_path, "config.json")  # Path to model configuration

# Ensure metadata.csv is generated for training
os.makedirs(dataset_path, exist_ok=True)

# Parse the .srt file and extract text
if os.path.exists(srt_file_path):
    subs = pysrt.open(srt_file_path)
    srt_data = {f"segment_{i+1}.wav": sub.text.replace('\n', ' ') for i, sub in enumerate(subs)}

    with open(metadata_path, "w") as f:
        for audio_file, text in srt_data.items():
            f.write(f"{audio_file}|{text}\n")

    # Display dataset information
    df = pd.DataFrame(list(srt_data.items()), columns=["Audio File", "Text"])
    print("Dataset Overview:")
    print(df.head())
else:
    print(f"Subtitle file {srt_file_path} not found!")

# Ensure model configuration exists
if not os.path.exists(config_path):
    print(f"Configuration file not found at {config_path}. Please ensure the correct config.json is available.")
    exit(1)

# Train the model using Coqui TTS Python API
train_command = f"python -m TTS.bin.train --config_path {config_path} --output_path fine_tuned_model --dataset_path {dataset_path} --epochs 50 --early_stopping True"
os.system(train_command)

# Load the fine-tuned model for inference
fine_tuned_model_path = "fine_tuned_model/best_model.pth"
tts = TTS(model_path=fine_tuned_model_path, config_path=config_path).to(device)

# Convert text to speech using fine-tuned model
text_to_speak = "This is a test using my fine-tuned model."
output_wav = "output.wav"
tts.tts_to_file(text=text_to_speak, file_path=output_wav)

print(f"Speech generated and saved at {output_wav}")