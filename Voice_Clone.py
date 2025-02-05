import os
import argparse
import torch
import pandas as pd
import torchaudio
from speechbrain.inference import Tacotron2, HIFIGAN, SpeakerRecognition
from speechbrain.dataio.encoder import TextEncoder
from torch.nn import L1Loss
from torch.optim import Adam

# Set Hugging Face authentication token
os.environ["HF_HOME"] = "./huggingface_cache"
os.environ["HF_TOKEN"] = "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"

# Define dataset path
data_dir = "/media/arupreza/Assets/LLM Projects/AccentFlow/Part_2/saved_output/extracted_audio_segments"
metadata_path = os.path.join(data_dir, "metadata.csv")

def load_text_data():
    """Loads text data from the metadata file."""
    if os.path.exists(metadata_path):
        try:
            df = pd.read_csv(metadata_path, sep='|', on_bad_lines='skip', engine='python', names=['filename', 'text'], skip_blank_lines=True)
            df = df.dropna(subset=['text'])
            text_data = df[['filename', 'text']].to_dict(orient='records')
            if not text_data:
                print("Error: Metadata file is empty or has no valid text entries.")
                return []
            return text_data
        except pd.errors.ParserError as e:
            print("Error parsing CSV file. Possible formatting issues:", e)
            return []
        except Exception as e:
            print("Unexpected error while reading metadata.csv:", e)
            return []
    else:
        print("Error: Metadata file not found!")
        return []

def train_model():
    """Trains the Tacotron2 model using the dataset."""
    text_data = load_text_data()
    if not text_data:
        print("Error: No valid training data found.")
        return
    
    # Define text tokenizer
    text_encoder = TextEncoder()
    text_encoder.update_from_iterable([sample["text"] for sample in text_data])
    
    # Load Tacotron2 in training mode
    tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tacotron2_model", use_auth_token=os.environ["HF_TOKEN"])
    optimizer = Adam(tacotron2.parameters(), lr=0.001)
    loss_fn = L1Loss()
    
    tacotron2.train()
    
    for epoch in range(10):  # Train for 10 epochs
        total_loss = 0
        for sample in text_data:
            filename = sample['filename']
            text = sample['text']
            audio_path = os.path.join(data_dir, filename)
            
            if os.path.exists(audio_path):
                print(f"Training on {filename} with text: {text}")
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Define Mel Spectrogram with correct shape
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=1024,
                    hop_length=256,
                    n_mels=80  # Ensure this matches Tacotron2's output
                )
                mel_spectrogram = mel_transform(waveform).squeeze(0).transpose(0, 1)  # Adjust shape
                
                # Ensure text is passed as a string
                text_str = text if isinstance(text, str) else " ".join(text_encoder.decode_sequence(text))
                
                optimizer.zero_grad()
                mel_outputs, mel_lengths, alignments = tacotron2.forward(text_str)
                
                # Trim or pad spectrograms to match sizes
                min_length = min(mel_outputs.shape[-1], mel_spectrogram.shape[-1])
                mel_outputs = mel_outputs[:, :, :min_length]
                mel_spectrogram = mel_spectrogram[:, :, :min_length]
                
                loss = loss_fn(mel_outputs, mel_spectrogram)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    
    print("Training complete! Model fine-tuned with dataset.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model with the dataset')
    args = parser.parse_args()
    
    if args.train:
        train_model()
    else:
        print("Error: Please specify --train to start training.")
    
if __name__ == "__main__":
    main()
