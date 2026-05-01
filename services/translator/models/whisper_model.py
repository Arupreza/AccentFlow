from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import subprocess
import tempfile
import os

class WhisperModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained("/app/checkpoints/whisper")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "/app/checkpoints/whisper"
        ).to(self.device)
        self.model.eval()

    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg → 16kHz mono WAV"""
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", tmp_wav
        ], check=True, capture_output=True)
        return tmp_wav

    def transcribe(self, audio_path: str) -> str:
        # If video, extract audio first
        if audio_path.endswith((".mp4", ".mkv", ".mov", ".avi")):
            audio_path = self._extract_audio(audio_path)

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(inputs.input_features)

        transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcript.strip()
