from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from schemas import GrammarRequest, CheckerRequest, TranslatorRequest
from models.whisper_model import WhisperModel
from models.grammar_model import GrammarModel
from models.checker_model import CheckerModel
from models.translator_model import TranslatorModel
import shutil, uuid, os

app = FastAPI()

STORAGE_DIR = "/app/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Load all models once at startup
whisper    = WhisperModel()
grammar    = GrammarModel()
checker    = CheckerModel()
translator = TranslatorModel()

# ───────── Health ─────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ───────── Text endpoints ─────────
@app.post("/correct")
def correct(req: GrammarRequest):
    return {"corrected": grammar.correct(req.text)}

@app.post("/check")
def check(req: CheckerRequest):
    return {"checked": checker.check(req.text)}

@app.post("/translate")
def translate(req: TranslatorRequest):
    return {"translated": translator.translate(req.text, req.source_lang, req.target_lang)}

# ───────── File upload helper ─────────
def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1]
    saved_path = os.path.join(STORAGE_DIR, f"{uuid.uuid4()}{ext}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return saved_path

# ───────── File upload endpoints ─────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    saved_path = _save_upload(file)
    try:
        transcript = whisper.transcribe(saved_path)
        return {"transcript": transcript, "saved_as": saved_path}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/pipeline")
async def pipeline(file: UploadFile = File(...), target_lang: str = Form("kor_Hang")):
    saved_path = _save_upload(file)
    try:
        transcript = whisper.transcribe(saved_path)
        corrected  = grammar.correct(transcript)
        checked    = checker.check(corrected)
        translated = translator.translate(checked, "eng_Latn", target_lang)
        return {
            "saved_as"   : saved_path,
            "transcript" : transcript,
            "corrected"  : corrected,
            "checked"    : checked,
            "translated" : translated
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
