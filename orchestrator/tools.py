import httpx
import os

TRANSLATOR_URL = os.getenv("TRANSLATOR_URL", "http://translator:8005")
TIMEOUT        = httpx.Timeout(300.0)


async def extract_audio(video_path: str) -> str:
    """Upload video, get extracted audio saved in shared storage."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        with open(video_path, "rb") as f:
            files = {"file": f}
            r = await client.post(f"{TRANSLATOR_URL}/extract_audio", files=files)
            r.raise_for_status()

            # Save audio response to storage
            audio_path = video_path.replace(".mp4", "_audio.wav")
            with open(audio_path, "wb") as out:
                out.write(r.content)
            return audio_path


async def transcribe(audio_path: str) -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Convert host path to container path (storage is shared)
        container_path = audio_path.replace("/app/storage", "/app/storage")
        r = await client.post(
            f"{TRANSLATOR_URL}/transcribe",
            json={"audio_path": container_path}
        )
        r.raise_for_status()
        return r.json()["transcript"]


async def correct_grammar(text: str) -> str:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(
            f"{TRANSLATOR_URL}/correct",
            json={"text": text}
        )
        r.raise_for_status()
        return r.json()["corrected"]


async def check_grammar(text: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(
            f"{TRANSLATOR_URL}/check",
            json={"text": text}
        )
        r.raise_for_status()
        return r.json()  # {"grammar_score": float, "is_acceptable": bool}