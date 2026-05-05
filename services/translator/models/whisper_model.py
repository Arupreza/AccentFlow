import gc
import torch
import librosa
import numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline,
)


class WhisperModel:
    def __init__(self, model_path: str = "/app/checkpoints/whisper-nf4"):
        self.model_path = model_path
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = None
        self.model = None
        self.pipe = None

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            clean_up_tokenization_spaces=True,
            local_files_only=True,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            quantization_config=self.bnb_config,
            device_map="auto",
            local_files_only=True,
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
        self._load()
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = audio.astype(np.float32)

            result = self.pipe(
                {"array": audio, "sampling_rate": sr},
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5,
            )

            del audio
            return result["text"]

        finally:
            # fires whether transcription succeeds or raises an exception
            self.unload()

    def unload(self):
        del self.pipe
        del self.model
        del self.processor
        self.pipe = None
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"VRAM freed — Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB"
            f" | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()