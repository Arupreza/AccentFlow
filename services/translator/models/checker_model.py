from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CheckerModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("/app/checkpoints/checker")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "/app/checkpoints/checker"
        ).to(self.device).eval()

    def check(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # CoLA: index 1 = "acceptable", index 0 = "unacceptable"
        score = torch.softmax(logits, dim=-1)[0][1].item()

        return {
            "grammar_score": round(score, 4),
            "is_acceptable": score > 0.5
        }