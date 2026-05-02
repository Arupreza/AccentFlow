# AccentFlow

> **Agentic AI pipeline that transforms accented English video into lip-synced, grammatically corrected, multi-language video — preserving the original speaker's voice.**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

AccentFlow takes a video of a non-native English speaker and produces a polished output where:

1. **Speech is transcribed** using OpenAI Whisper
2. **Grammar is corrected** using Grammarly's CoEdit-Large
3. **Grammatical quality is scored** using BERT-CoLA
4. **Text is translated** to target language using Meta's NLLB-200
5. **Voice is cloned** to speak the new text in the original speaker's voice (Fish-Speech S2-Pro)
6. **Lips are re-synced** to match the new audio (ByteDance LatentSync)

Result: a video where the original speaker appears to fluently speak corrected/translated content in their own voice.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AccentFlow Microservices                    │
└─────────────────────────────────────────────────────────────────┘

         ┌──────────────────────┐
         │     Orchestrator     │  port 8000
         │   (LangGraph Agent)  │  no GPU
         └──────────┬───────────┘
                    │
        ┌───────────┼───────────────────┐
        │           │                   │
        ▼           ▼                   ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ Translator  │ │  Fish-TTS    │ │ LatentSync   │
│  port 8005  │ │  port 8003   │ │  port 8004   │
│  CUDA 11.8  │ │  CUDA 11.8   │ │  CUDA 11.8   │
├─────────────┤ ├──────────────┤ ├──────────────┤
│ Whisper     │ │ S2-Pro       │ │ LatentSync   │
│ CoEdit      │ │ DualAR       │ │ UNet 5GB     │
│ CoLA        │ │ DAC Decoder  │ │ SyncNet      │
│ NLLB-200    │ │              │ │ Aux models   │
└─────────────┘ └──────────────┘ └──────────────┘
        │           │                   │
        └───────────┼───────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │    Shared Storage    │
        │   (host volume)      │
        └──────────────────────┘
```

Each service runs in its own Docker container with isolated CUDA runtime, independent scaling, and clean failure boundaries.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Models** | Whisper-Large-v3-Turbo, CoEdit-Large, BERT-CoLA, NLLB-200, Fish-Speech S2-Pro, LatentSync v1.6 |
| **Frameworks** | PyTorch 2.1, Transformers 4.44, FastAPI, LangGraph |
| **Infrastructure** | Docker Compose, NVIDIA Container Toolkit, CUDA 11.8 |
| **Language** | Python 3.10 |
| **API Format** | REST (JSON) for translator/orchestrator, MessagePack for Fish-TTS |

---

## Prerequisites

### Hardware
- **GPU:** NVIDIA with ≥ 16GB VRAM (24GB+ recommended for running all services together)
- **RAM:** 32GB+ system memory
- **Storage:** 80GB+ for models and Docker images

### Software
- Ubuntu 20.04 / 22.04
- Docker Engine 24.0+
- Docker Compose v2.20+
- NVIDIA Driver ≥ 535.xx
- NVIDIA Container Toolkit

### Verify Prerequisites
```bash
# GPU & driver
nvidia-smi

# Docker
docker --version
docker compose version

# NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Project Structure

```
AccentFlow-0.2/
│
├── checkpoints/              ← Model weights (host-only, not committed)
│   ├── whisper/              Whisper-Large-v3-Turbo
│   ├── grammarly/            CoEdit-Large
│   ├── checker/              BERT-CoLA (grammar scorer)
│   ├── translator/           NLLB-200-distilled-600M
│   ├── fish_tts/             Fish-Speech S2-Pro
│   └── latentsync/           LatentSync v1.6
│
├── storage/                  ← Shared runtime files (videos, audio)
│
├── services/
│   ├── translator/
│   │   ├── Dockerfile
│   │   ├── api.py
│   │   ├── schemas.py
│   │   ├── requirements.txt
│   │   └── models/
│   │       ├── whisper_model.py
│   │       ├── grammar_model.py
│   │       ├── checker_model.py
│   │       └── translator_model.py
│   │
│   ├── fish_tts/
│   │   └── Dockerfile        ← Runs Fish-Speech's official server
│   │
│   └── latentsync/
│       ├── Dockerfile
│       ├── api.py
│       └── requirements.txt
│
├── orchestrator/             ← LangGraph multi-agent pipeline
│   ├── Dockerfile
│   ├── main.py
│   ├── agent.py
│   └── state.py
│
├── shared/
│   └── schemas.py
│
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/Arupreza/AccentFlow.git
cd AccentFlow
git checkout AccentFlow-Beta
```

### 2. Download Model Checkpoints

Models are NOT committed to Git (too large). Download separately:

```bash
# Authenticate with HuggingFace (required for some models)
huggingface-cli login

# Whisper-Large-v3-Turbo
huggingface-cli download openai/whisper-large-v3-turbo \
    --local-dir checkpoints/whisper \
    --include "*.safetensors" "*.json" "*.txt"

# Grammar correction
huggingface-cli download grammarly/coedit-large \
    --local-dir checkpoints/grammarly \
    --include "*.safetensors" "*.json" "*.txt" "*.model"

# Grammar scoring (CoLA)
huggingface-cli download textattack/bert-base-uncased-CoLA \
    --local-dir checkpoints/checker \
    --include "*.bin" "*.safetensors" "*.json" "*.txt"

# Translation
huggingface-cli download facebook/nllb-200-distilled-600M \
    --local-dir checkpoints/translator \
    --include "*.safetensors" "*.bin" "*.json" "*.txt" "*.model"

# Fish-TTS S2-Pro (gated — request access first at huggingface.co/fishaudio/s2-pro)
huggingface-cli download fishaudio/s2-pro \
    --local-dir checkpoints/fish_tts \
    --include "*.safetensors" "*.json" "*.pth" "*.jinja" "*.model" "*.txt"

# LatentSync v1.6
huggingface-cli download ByteDance/LatentSync-1.6 \
    --local-dir checkpoints/latentsync \
    --include "*.pt" "*.pth" "*.bin" "*.safetensors" "*.json"
```

**Total download size: ~25 GB**. Use `nohup` for long downloads over SSH:
```bash
nohup huggingface-cli download <repo> --local-dir <path> > download.log 2>&1 &
tail -f download.log
```

### 3. Build Docker Containers

```bash
# Build all services
docker compose build

# Or build individually (recommended for first time)
docker compose build translator     # ~15 min
docker compose build fish_tts       # ~25 min
docker compose build latentsync     # ~30 min
```

### 4. Start Services

```bash
# Start individually based on what you need
docker compose up -d translator
docker compose up -d fish_tts
docker compose up -d latentsync

# Or start everything together (requires 24GB+ VRAM)
docker compose up -d

# Verify
docker ps
```

### 5. Health Checks

```bash
curl http://localhost:8005/health    # Translator
curl http://localhost:8003/v1/health 2>&1 | head -5  # Fish-TTS
curl http://localhost:8004/health    # LatentSync
```

---

## API Reference

### Translator Service (port 8005)

#### `POST /transcribe`
Transcribe audio/video file to text using Whisper.

```python
import requests

r = requests.post("http://localhost:8005/transcribe", json={
    "audio_path": "/app/storage/input.mp4"
})
# Response: {"transcript": "..."}
```

#### `POST /correct`
Fix grammar using CoEdit-Large.

```python
r = requests.post("http://localhost:8005/correct", json={
    "text": "He don't like apples"
})
# Response: {"corrected": "He doesn't like apples."}
```

#### `POST /check`
Score grammatical acceptability using BERT-CoLA.

```python
r = requests.post("http://localhost:8005/check", json={
    "text": "She goes to school every day."
})
# Response: {"grammar_score": 0.94, "is_acceptable": true}
```

#### `POST /translate`
Translate text using NLLB-200.

```python
r = requests.post("http://localhost:8005/translate", json={
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "kor_Hang"
})
# Response: {"translated": "안녕하세요, 어떻게 지내세요?"}
```

**Supported language codes:**

| Language | Code | Language | Code |
|---|---|---|---|
| English | eng_Latn | Korean | kor_Hang |
| Japanese | jpn_Jpan | Chinese | zho_Hans |
| Bengali | ben_Beng | Spanish | spa_Latn |
| French | fra_Latn | German | deu_Latn |
| Hindi | hin_Deva | Arabic | arb_Arab |

#### `POST /extract_audio`
Extract audio track from video file (returns WAV).

```python
with open("video.mp4", "rb") as f:
    r = requests.post(
        "http://localhost:8005/extract_audio",
        files={"file": f}
    )

with open("audio.wav", "wb") as out:
    out.write(r.content)
```

---

### Fish-TTS Service (port 8003)

Uses Fish-Speech's official MessagePack API.

#### `POST /v1/tts`
Synthesize speech with optional voice cloning.

```python
import requests
import ormsgpack

# Voice cloning (recommended)
with open("reference.wav", "rb") as f:
    ref_audio = f.read()

payload = {
    "text": "Text to be synthesized",
    "references": [{
        "audio": ref_audio,
        "text": "Exact transcript of reference audio"
    }],
    "format": "wav",
    "max_new_tokens": 1024,
    "chunk_length": 200,
    "top_p": 0.7,
    "repetition_penalty": 1.2,
    "temperature": 0.7,
    "streaming": False
}

r = requests.post(
    "http://localhost:8003/v1/tts",
    headers={"content-type": "application/msgpack"},
    data=ormsgpack.packb(payload),
    timeout=600
)

with open("output.wav", "wb") as f:
    f.write(r.content)
```

**Reference audio requirements:**
- Format: 22050 Hz mono WAV
- Duration: 5–30 seconds
- Quality: Clean speech, minimal background noise
- Reference text MUST match audio exactly (use `/transcribe` to get accurate text)

---

### LatentSync Service (port 8004)

#### `POST /sync`
Generate lip-synced video from input video + audio.

```python
import requests

with open("video.mp4", "rb") as v, open("audio.wav", "rb") as a:
    r = requests.post(
        "http://localhost:8004/sync",
        files={"video": v, "audio": a},
        timeout=900
    )

with open("synced_video.mp4", "wb") as f:
    f.write(r.content)
```

**Requirements:**
- Front-facing visible face in video
- Audio length ≥ video length
- Recommended video duration: 5–30 seconds
- VRAM: ~12 GB during inference

---

## Full Pipeline Example

End-to-end usage in Jupyter:

```python
import requests
import ormsgpack
import torchaudio

# ─── Setup ───
INPUT_VIDEO = "/path/to/input.mp4"
TARGET_LANG = "kor_Hang"

# ─── 1. Extract audio from input video ───
with open(INPUT_VIDEO, "rb") as f:
    r = requests.post(
        "http://localhost:8005/extract_audio",
        files={"file": f}
    )
with open("storage/extracted.wav", "wb") as out:
    out.write(r.content)

# ─── 2. Resample reference audio for Fish-TTS ───
waveform, sr = torchaudio.load("storage/extracted.wav")
if sr != 22050:
    waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
torchaudio.save("storage/reference_22k.wav", waveform[:, :22050*20], 22050)

# ─── 3. Transcribe ───
r = requests.post("http://localhost:8005/transcribe",
    json={"audio_path": "/app/storage/reference_22k.wav"})
transcript = r.json()["transcript"]
print("Original:", transcript)

# ─── 4. Correct grammar ───
r = requests.post("http://localhost:8005/correct", json={"text": transcript})
corrected = r.json()["corrected"]
print("Corrected:", corrected)

# ─── 5. Score quality ───
r = requests.post("http://localhost:8005/check", json={"text": corrected})
print("Grammar score:", r.json())

# ─── 6. Translate ───
r = requests.post("http://localhost:8005/translate", json={
    "text": corrected,
    "source_lang": "eng_Latn",
    "target_lang": TARGET_LANG
})
translated = r.json()["translated"]
print("Translated:", translated)

# ─── 7. Voice clone synthesis ───
with open("storage/reference_22k.wav", "rb") as f:
    ref_audio = f.read()

payload = {
    "text": translated,
    "references": [{"audio": ref_audio, "text": transcript}],
    "format": "wav",
    "max_new_tokens": 1024,
    "chunk_length": 200,
    "top_p": 0.7,
    "temperature": 0.7,
    "streaming": False
}
r = requests.post(
    "http://localhost:8003/v1/tts",
    headers={"content-type": "application/msgpack"},
    data=ormsgpack.packb(payload),
    timeout=600
)
with open("storage/cloned_voice.wav", "wb") as f:
    f.write(r.content)

# ─── 8. Lip-sync ───
with open(INPUT_VIDEO, "rb") as v, open("storage/cloned_voice.wav", "rb") as a:
    r = requests.post(
        "http://localhost:8004/sync",
        files={"video": v, "audio": a},
        timeout=900
    )
with open("final_output.mp4", "wb") as f:
    f.write(r.content)

print("Done. Output: final_output.mp4")
```

---

## Performance & Resource Requirements

| Service | VRAM | First inference | Subsequent | Disk |
|---|---|---|---|---|
| Translator | ~8 GB | ~30s | ~3-10s | ~12 GB |
| Fish-TTS | ~10 GB | ~60s | ~5-15s | ~12 GB |
| LatentSync | ~12 GB | ~5min | ~30s-3min | ~10 GB |

**Running all 3 simultaneously requires 24 GB+ VRAM.** On smaller GPUs, run sequentially.

---

## Common Issues

### CUDA Out of Memory
```bash
# Stop one service to free VRAM
docker compose stop translator
nvidia-smi   # verify freed
docker compose up -d fish_tts
```

### Container Won't Start
```bash
# View logs to diagnose
docker compose logs <service_name> --tail 50

# Common causes:
# - Empty Python files (VS Code save issue) — verify with: ls -lh services/<name>/
# - Missing model checkpoints — verify with: ls checkpoints/<name>/
# - Port conflict — check: ss -tlnp | grep 8005
```

### Model Download Fails
```bash
# Re-authenticate
huggingface-cli logout
huggingface-cli login

# Resume interrupted download (huggingface-cli auto-resumes)
nohup huggingface-cli download <repo> --local-dir <path> > download.log 2>&1 &
```

### Fish-TTS Decoder Config Error
```
Use --decoder-config-name modded_dac_vq, NOT firefly_gan_vq.
The firefly_gan_vq config is from older Fish-Speech v1.4 docs.
v2.0.0+ uses modded_dac_vq for S2-Pro architecture.
```

---

## Development

### Adding a New Service

1. Create `services/<name>/` directory with Dockerfile, api.py, requirements.txt
2. Add service block to `docker-compose.yml`
3. Mount required checkpoint volumes
4. Build & test:
   ```bash
   docker compose build <name>
   docker compose up -d <name>
   ```

### Modifying API Code

```bash
# Code is baked into image via COPY — rebuild required
docker compose down <service>
docker compose build <service>
docker compose up -d <service>
```

### Updating Checkpoints

Checkpoints are mounted as volumes — no rebuild needed:
```bash
# Replace files in checkpoints/<name>/
docker compose restart <service>
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

This project uses third-party models with their own licenses:
- Whisper (MIT)
- CoEdit-Large (CC BY-NC 4.0 — non-commercial)
- BERT-CoLA (MIT)
- NLLB-200 (CC BY-NC 4.0 — non-commercial)
- Fish-Speech (CC BY-NC-SA 4.0 — non-commercial)
- LatentSync (Apache 2.0)

**Note:** Several models are non-commercial. For commercial deployment, replace with appropriate alternatives.

---

## Acknowledgments

- OpenAI for Whisper
- Grammarly for CoEdit
- Meta AI for NLLB-200
- Fish Audio for Fish-Speech S2-Pro
- ByteDance for LatentSync
- HuggingFace for the model hub

---

## Contact

**Md Rezanur Islam (Reza)**
LLM Engineer & Agentic AI Developer
PhD Candidate, Soonchunhyang University

- Website: [a2zai.xyz](https://a2zai.xyz)
- GitHub: [@Arupreza](https://github.com/Arupreza)

---

## Citation

If you use this work in research:

```bibtex
@software{accentflow2026,
  author  = {Islam, Md Rezanur},
  title   = {AccentFlow: Agentic AI Pipeline for Accent-Adaptive Video Synthesis},
  year    = {2026},
  url     = {https://github.com/Arupreza/AccentFlow}
}
```