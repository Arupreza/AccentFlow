import os
import torch
from TTS.api import TTS
from TTS.trainer import Trainer, TrainingArgs

# Ensure GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download and load the pre-trained lightweight model
tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(tts_model).to(device)

# Define dataset path (Ensure your dataset is structured properly)
dataset_path = "dataset"  # Folder containing metadata.csv and wavs/

# Define training parameters
training_args = TrainingArgs(
    restore_path=None,  # Set None to start fresh or specify a pre-trained model checkpoint
    output_path="fine_tuned_model",  # Where to save fine-tuned model
    dataset_path=dataset_path,  # Path to your dataset
    config_path=tts.config_path,  # Use default config
    batch_size=16,  # Adjust based on available GPU memory
    num_epochs=50,  # Number of epochs
    use_cuda=(device == "cuda"),  # Enable CUDA if available
)

# Initialize trainer and start training
trainer = Trainer(training_args)
trainer.fit()

# Save the fine-tuned model
fine_tuned_model_path = "fine_tuned_model/my_finetuned_tts.pth"
torch.save(trainer.model.state_dict(), fine_tuned_model_path)
print(f"Fine-tuned model saved successfully at {fine_tuned_model_path}")

# Load the fine-tuned model for inference
tts.load_checkpoint(fine_tuned_model_path)

# Convert text to speech using fine-tuned model
text_to_speak = "This is a test using my fine-tuned model."
output_wav = "output.wav"
tts.tts_to_file(text=text_to_speak, file_path=output_wav)

print(f"Speech generated and saved at {output_wav}")