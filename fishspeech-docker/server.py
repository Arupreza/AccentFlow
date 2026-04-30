import os
import subprocess
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

app = FastAPI()

FISH_DIR   = "/app/fish-speech"
MODEL_PATH = "/app/fish-speech/models/s2-pro"
OUTPUTS    = "/outputs"
os.makedirs(OUTPUTS, exist_ok=True)


@app.post("/tts")
async def tts(
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
    output_filename: str = Form("output.wav"),
):
    with tempfile.TemporaryDirectory() as tmp:
        # Save uploaded reference audio
        ref_audio_path = f"{tmp}/ref.wav"
        with open(ref_audio_path, "wb") as f:
            f.write(await ref_audio.read())

        # Stage 1: encode reference voice
        subprocess.run([
            "python", f"{FISH_DIR}/fish_speech/models/dac/inference.py",
            "-i", ref_audio_path,
            "--checkpoint-path", f"{MODEL_PATH}/codec.pth",
        ], cwd=FISH_DIR, check=True)

        # Stage 2: text → semantic tokens
        subprocess.run([
            "python", f"{FISH_DIR}/fish_speech/models/text2semantic/inference.py",
            "--text", text,
            "--prompt-text", ref_text,
            "--prompt-tokens", f"{FISH_DIR}/fake.npy",
            "--checkpoint-path", MODEL_PATH,
        ], cwd=FISH_DIR, check=True)

        # Stage 3: semantic tokens → audio
        output_path = f"{OUTPUTS}/{output_filename}"
        subprocess.run([
            "python", f"{FISH_DIR}/fish_speech/models/dac/inference.py",
            "--codes", f"{FISH_DIR}/codes_0",
            "--output", output_path,
            "--checkpoint-path", f"{MODEL_PATH}/codec.pth",
        ], cwd=FISH_DIR, check=True)

    return FileResponse(output_path, media_type="audio/wav", filename=output_filename)


@app.get("/health")
def health():
    return {"status": "ok"}