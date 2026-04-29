#!/bin/bash
set -e

VIDEO=${VIDEO_PATH:-/data/input/video.mp4}
AUDIO=${AUDIO_PATH:-/data/input/audio.wav}
OUTPUT=${OUTPUT_PATH:-/data/output/result.mp4}
CKPT=${CKPT_PATH:-/data/checkpoints/latentsync_unet.pt}
CONFIG=${CONFIG_PATH:-configs/unet/second_stage.yaml}
STEPS=${INFERENCE_STEPS:-20}
GUIDANCE=${GUIDANCE_SCALE:-1.5}
SEED=${SEED:-1247}

if [[ ! -f "$VIDEO" ]]; then echo "ERROR: video not found at $VIDEO"; exit 1; fi
if [[ ! -f "$AUDIO" ]]; then echo "ERROR: audio not found at $AUDIO"; exit 1; fi
if [[ ! -f "$CKPT"  ]]; then echo "ERROR: checkpoint not found at $CKPT"; exit 1; fi

mkdir -p /data/output

exec python -m scripts.inference \
    --unet_config_path "$CONFIG" \
    --inference_ckpt_path "$CKPT" \
    --video_path "$VIDEO" \
    --audio_path "$AUDIO" \
    --video_out_path "$OUTPUT" \
    --inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE" \
    --seed "$SEED" \
    "$@"