from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class CheckerModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("/app/checkpoints/checker")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "/app/checkpoints/checker"
        ).to(self.device)
        self.model.eval()

    def check(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
