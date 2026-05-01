from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class GrammarModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("/app/checkpoints/grammarly")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "/app/checkpoints/grammarly"
        ).to("cuda")

    def correct(self, text: str) -> str:
        inputs = self.tokenizer(
            f"Fix grammatical errors: {text}",
            return_tensors="pt"
        ).to("cuda")
        output = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)