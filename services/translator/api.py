from fastapi import FastAPI
from schemas import TranscribeRequest, GrammarRequest, CheckerRequest, TranslatorRequest, PipelineRequest
from models.whisper_model import WhisperModel
from models.grammar_model import GrammarModel
from models.checker_model import CheckerModel
from models.translator_model import TranslatorModel

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
    return {"transcript": whisper.transcribe(req.audio_path)}

@app.post("/correct")
def correct(req: GrammarRequest):
    return {"corrected": grammar.correct(req.text)}

@app.post("/check")
def check(req: CheckerRequest):
    return {"checked": checker.check(req.text)}

@app.post("/translate")
def translate(req: TranslatorRequest):
    return {"translated": translator.translate(req.text, req.source_lang, req.target_lang)}

@app.post("/pipeline")
def pipeline(req: PipelineRequest):
    transcript = whisper.transcribe(req.audio_path)
    corrected  = grammar.correct(transcript)
    checked    = checker.check(corrected)
    translated = translator.translate(checked, "eng_Latn", req.target_lang)
    return {
        "transcript" : transcript,
        "corrected"  : corrected,
        "checked"    : checked,
        "translated" : translated
    }
