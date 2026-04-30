#!/bin/bash
set -e

TEXT=${TEXT:-""}
REFERENCE_AUDIO=${REFERENCE_AUDIO:-""}
REFERENCE_TEXT=${REFERENCE_TEXT:-""}
LLAMA_CKPT=${LLAMA_CKPT:-/data/checkpoints/openaudio-s1-mini}
DECODER_CKPT=${DECODER_CKPT:-/data/checkpoints/openaudio-s1-mini/codec.pth}
OUTPUT=${OUTPUT_PATH:-/data/output/result.wav}
WORKDIR=/tmp/fish_workdir
COMPILE=${COMPILE:-0}
HALF=${HALF:-0}

if [[ -z "$TEXT" ]]; then echo "ERROR: TEXT env var is required"; exit 1; fi
if [[ ! -d "$LLAMA_CKPT" ]]; then echo "ERROR: checkpoint not found at $LLAMA_CKPT"; exit 1; fi
if [[ ! -f "$DECODER_CKPT" ]]; then echo "ERROR: codec not found at $DECODER_CKPT"; exit 1; fi

mkdir -p "$WORKDIR" /data/output

# Stage 1: VQ encode reference audio (voice cloning only)
if [[ -n "$REFERENCE_AUDIO" ]]; then
    echo "[1/3] Encoding reference audio..."
    python fish_speech/models/dac/inference.py \
        -i "$REFERENCE_AUDIO" \
        --checkpoint-path "$DECODER_CKPT" \
        --output-path "$WORKDIR/fake.npy"
    PROMPT_TOKENS="$WORKDIR/fake.npy"
else
    echo "[1/3] Skipping VQ encoding (random voice)"
    PROMPT_TOKENS=""
fi

# Stage 2: Text to semantic tokens
echo "[2/3] Generating semantic tokens..."
T2S_ARGS=(
    --text "$TEXT"
    --checkpoint-path "$LLAMA_CKPT"
    --output-path "$WORKDIR/codes"
)
[[ -n "$PROMPT_TOKENS" ]]  && T2S_ARGS+=(--prompt-tokens "$PROMPT_TOKENS")
[[ -n "$REFERENCE_TEXT" ]] && T2S_ARGS+=(--prompt-text "$REFERENCE_TEXT")
[[ "$COMPILE" == "1" ]]    && T2S_ARGS+=(--compile)
[[ "$HALF" == "1" ]]       && T2S_ARGS+=(--half)

python fish_speech/models/text2semantic/inference.py "${T2S_ARGS[@]}"

# Stage 3: Semantic tokens to audio
echo "[3/3] Synthesizing audio..."
python fish_speech/models/dac/inference.py \
    -i "$WORKDIR/codes_0.npy" \
    --checkpoint-path "$DECODER_CKPT" \
    --output-path "$OUTPUT"

echo "Done → $OUTPUT"