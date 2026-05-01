from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TranslatorModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/app/checkpoints/translator")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "/app/checkpoints/translator"
        ).to("cuda")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang)
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)