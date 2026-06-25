import gc
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

MAX_SENTENCES_PER_BATCH = 5


class GrammarModel:
    def __init__(self, model_path: str = "/app/checkpoints/grammarly/full"):
        ckpt = Path(model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
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
        self._warmup()

    def _warmup(self):
        dummy = self.tokenizer(
            "Fix grammar: warmup.",
            return_tensors="pt",
            max_length=32,
            truncation=True,
        ).to("cuda")
        with torch.no_grad():
            self.model.generate(**dummy, max_new_tokens=8, num_beams=1, do_sample=False)
        del dummy
        torch.cuda.empty_cache()

    def _correct_batch(self, sentences: list[str]) -> list[str]:
        """Run one generate() call on a small batch of sentences."""
        prompts = [f"Fix grammar: {s}" for s in sentences]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
        ).to("cuda")

        input_token_count = inputs["input_ids"].shape[1]
        dynamic_max_new_tokens = min(int(input_token_count * 1.2), 128)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=dynamic_max_new_tokens,
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del inputs, outputs
        torch.cuda.empty_cache()

        return decoded

    def correct(self, text: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        non_empty = [(i, s) for i, s in enumerate(sentences) if s.strip()]
        if not non_empty:
            return text

        indices, valid_sentences = zip(*non_empty)

        corrected_sentences = []
        for i in range(0, len(valid_sentences), MAX_SENTENCES_PER_BATCH):
            batch = valid_sentences[i: i + MAX_SENTENCES_PER_BATCH]
            corrected_sentences.extend(self._correct_batch(list(batch)))

        result_map = dict(zip(indices, corrected_sentences))
        final = [result_map.get(i, sentences[i]) for i in range(len(sentences))]

        gc.collect()
        torch.cuda.empty_cache()

        return " ".join(final)

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