from pydantic import BaseModel

class TranscribeRequest(BaseModel):
    audio_path: str

class GrammarRequest(BaseModel):
    text: str

class CheckerRequest(BaseModel):
    text: str

class TranslatorRequest(BaseModel):
    text: str
    source_lang: str = "eng_Latn"
    target_lang: str = "kor_Hang"