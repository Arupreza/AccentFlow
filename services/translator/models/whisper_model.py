import whisper

class WhisperModel:
    def __init__(self):
        self.model = whisper.load_model(
            "/app/checkpoints/whisper/model.safetensors"
        )

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path)
        return result["text"]