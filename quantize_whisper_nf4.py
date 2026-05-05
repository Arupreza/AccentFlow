"""
quantize_whisper_nf4.py
-----------------------
Quantizes a local Whisper checkpoint to NF4 (bitsandbytes) and saves all
artifacts needed for Docker cold-start (no HuggingFace download required).

Requirements:
    pip install transformers bitsandbytes>=0.43.0 accelerate safetensors torch

Run on the HOST machine (GPU required):
    python quantize_whisper_nf4.py
"""

import json
import shutil
from pathlib import Path

import torch
from transformers import (
    BitsAndBytesConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_CKPT = Path("/home/lisa/Arupreza/AccentFlow-0.2/checkpoints/whisper")
DST_CKPT = Path("/home/lisa/Arupreza/AccentFlow-0.2/checkpoints/whisper-nf4")

DST_CKPT.mkdir(parents=True, exist_ok=True)

# ── NF4 config ─────────────────────────────────────────────────────────────
# double_quant: quantize the quantization constants themselves (saves ~0.4 bpw)
# compute_dtype float16: matmuls dequantized to fp16 on the fly (faster than bf16 on most Whisper GPU setups)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"[1/4] Loading Whisper from: {SRC_CKPT}")
model = WhisperForConditionalGeneration.from_pretrained(
    str(SRC_CKPT),
    quantization_config=bnb_config,
    device_map="auto",           # spreads across available GPUs / CPU offload
    low_cpu_mem_usage=True,      # stream weights — avoids OOM on large variants
)

# Disable cache for encoder-decoder (required for proper NF4 save)
model.config.use_cache = False

# ── Save quantized model ────────────────────────────────────────────────────
print(f"[2/4] Saving NF4 model to: {DST_CKPT}")
model.save_pretrained(
    str(DST_CKPT),
    safe_serialization=True,     # saves as .safetensors (not .bin) — faster Docker COPY & load
)

# ── Save processor (tokenizer + feature extractor) ─────────────────────────
print("[3/4] Saving processor (tokenizer + feature extractor)...")
processor = WhisperProcessor.from_pretrained(str(SRC_CKPT))
processor.save_pretrained(str(DST_CKPT))

# ── Write a metadata sidecar ────────────────────────────────────────────────
# Docker entrypoint reads this to skip HF download and validate checkpoint.
meta = {
    "quant_type": "nf4",
    "double_quant": True,
    "compute_dtype": "float16",
    "source_checkpoint": str(SRC_CKPT),
    "bitsandbytes_min_version": "0.43.0",
    "safe_serialization": True,
    "note": "Load with BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)",
}
with open(DST_CKPT / "quant_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("[4/4] Done.")
print(f"\nSaved artifacts:")
for p in sorted(DST_CKPT.iterdir()):
    size = p.stat().st_size / 1e6
    print(f"  {p.name:50s}  {size:.1f} MB")