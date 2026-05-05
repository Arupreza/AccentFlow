from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from schemas import TranscribeRequest, GrammarRequest, CheckerRequest, TranslatorRequest
from models.whisper_model import WhisperModel
from models.grammar_model import GrammarModel
from models.checker_model import CheckerModel
from models.translator_model import TranslatorModel
import shutil, uuid, subprocess, os

app = FastAPI()

# Load all models once at startup
whisper    = WhisperModel()
grammar    = GrammarModel()
checker    = CheckerModel()
translator = TranslatorModel()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {req.audio_path}")
    try:
        transcript = whisper.transcribe(req.audio_path)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/correct")
def correct(req: GrammarRequest):
    return {"corrected": grammar.correct(req.text)}


@app.post("/check")
def check(req: CheckerRequest):
    return checker.check(req.text)         # ← changed: returns dict directly


@app.post("/translate")
def translate(req: TranslatorRequest):
    return {"translated": translator.translate(req.text, req.source_lang, req.target_lang)}


@app.post("/extract_audio")
async def extract_audio(file: UploadFile = File(...)):
    """
    Input  : video file (uploaded)
    Output : extracted audio (16kHz mono WAV)
    """
    job_id     = str(uuid.uuid4())
    video_path = f"/app/storage/{job_id}.mp4"
    audio_path = f"/app/storage/{job_id}.wav"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-vn", audio_path
    ], check=True, capture_output=True)

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{job_id}.wav"
    )