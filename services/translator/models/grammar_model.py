import gc
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


class GrammarModel:
    def __init__(self, model_path: str = "/app/checkpoints/grammarly/quantized"):
        ckpt = Path(model_path)

        with open(ckpt / "quant_meta.json") as f:
            meta = json.load(f)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=meta["quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, meta["compute_dtype"]),
            bnb_4bit_use_double_quant=meta["double_quant"],
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt),
            local_files_only=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            str(ckpt),
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()

    def _correct_sentence(self, sentence: str) -> str:
        if not sentence.strip():
            return sentence

        inputs = self.tokenizer(
            f"Fix grammatical errors in this sentence: {sentence}",
            return_tensors="pt",
            max_length=256,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def correct(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        corrected = [self._correct_sentence(s) for s in sentences]
        result = " ".join(corrected)

        gc.collect()
        torch.cuda.empty_cache()

        return result

    def unload(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"VRAM freed — Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB"
            f" | Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()