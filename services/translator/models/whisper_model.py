import torch
import librosa
import numpy as np
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperModel:
    def __init__(self, model_path: str = "/app/checkpoints/whisper"):
        # Explicitly define the parameter to suppress the warning
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            clean_up_tokenization_spaces=True
        )
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def transcribe(self, audio_path: str) -> str:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)

        result = self.pipe(
            {"array": audio, "sampling_rate": sr},
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5,
        )

        # clear intermediate tensors after each call
        del audio
        gc.collect()
        torch.cuda.empty_cache()

        return result["text"]

    def unload(self):
        """Call when completely done with the model."""
        del self.pipe
        del self.model
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VRAM freed — Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # context manager support — auto unloads on exit
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()