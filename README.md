# AccentFlow

> **Agentic AI pipeline that transforms accented English video into lip-synced, grammatically corrected, multi-language video — preserving the original speaker's voice. Powered by a LangGraph reflection agent that self-validates output quality.**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Reflection-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

AccentFlow takes a video of a non-native English speaker and produces a polished output where:

1. **Speech is transcribed** using OpenAI Whisper-Large-v3-Turbo
2. **Grammar is corrected** using Grammarly's CoEdit-Large
3. **Grammatical quality is scored** using BERT-CoLA (reflection loop)
4. **Text is translated** to target language using Meta's NLLB-200
5. **Voice is cloned** to speak new text in original speaker's voice (Fish-Speech S2-Pro)
6. **Lips are re-synced** to match new audio (MuseTalk v1.5)

A LangGraph-based reflection agent orchestrates the pipeline, automatically retrying grammar correction until quality threshold (≥ 0.9 score) is met.

Result: a video where the original speaker appears to fluently speak corrected/translated content in their own voice.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      AccentFlow Microservices                        │
└──────────────────────────────────────────────────────────────────────┘

         ┌────────────────────────────────┐
         │       Orchestrator             │  port 8000
         │   (LangGraph Reflection Agent) │  no GPU
         │                                │
         │   - Workflow state management  │
         │   - Reflection loop control    │
         │   - Service coordination       │
         └─────────────┬──────────────────┘
                       │
        ┌──────────────┼─────────────────┐
        │              │                 │
        ▼              ▼                 ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│ Translator  │  │  Fish-TTS    │  │  MuseTalk    │
│  port 8005  │  │  port 8003   │  │  port 8004   │
│  CUDA 11.8  │  │  CUDA 11.8   │  │  CUDA 11.8   │
├─────────────┤  ├──────────────┤  ├──────────────┤
│ Whisper     │  │ S2-Pro       │  │ MuseTalk v1.5│
│ CoEdit      │  │ DualAR       │  │ UNet 3.4GB   │
│ CoLA        │  │ DAC Decoder  │  │ SD-VAE       │
│ NLLB-200    │  │              │  │ DWPose       │
│             │  │              │  │ Face-Parse   │
└─────────────┘  └──────────────┘  └──────────────┘
        │              │                 │
        └──────────────┼─────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │      Shared Storage          │
        │   (host volume mount)        │
        └──────────────────────────────┘
```

Each service runs in its own Docker container with isolated CUDA runtime, independent scaling, and clean failure boundaries.

---

## Reflection Agent Flow

The orchestrator implements a self-improving pipeline using LangGraph's conditional state machine:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Agent Pipeline (LangGraph)                      │
└──────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   Video Input   │
                        │   (.mp4 file)   │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Extract Audio   │  → translator
                        │ (ffmpeg)        │   /extract_audio
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   Transcribe    │  → translator
                        │   (Whisper)     │   /transcribe
                        └────────┬────────┘
                                 │
                                 ▼
                ┌─────────────────────────────┐
                │      Correct Grammar        │  ◄────────┐
                │      (CoEdit-Large)         │           │
                │      iteration += 1         │           │
                └────────────┬────────────────┘           │
                             │                            │
                             ▼                            │
                ┌─────────────────────────────┐           │
                │     Check Grammar Score     │           │
                │     (BERT-CoLA)             │           │
                │     score = softmax(logits) │           │
                └────────────┬────────────────┘           │
                             │                            │
                             ▼                            │
                      ┌──────────────┐                    │
                      │  Reflection  │                    │
                      │   Decision   │                    │
                      └──────┬───────┘                    │
                             │                            │
                ┌────────────┼────────────┐               │
                │            │            │               │
                ▼            ▼            ▼               │
         ┌──────────┐ ┌──────────┐ ┌──────────┐           │
         │ score    │ │ score    │ │ iter ≥   │           │
         │ ≥ 0.9    │ │ < 0.9    │ │ max      │           │
         │  ✓       │ │  retry   │ │  ✗ stop  │           │
         └────┬─────┘ └────┬─────┘ └────┬─────┘           │
              │            │            │                 │
              │            └────────────┼─────────────────┘
              │                         │   (loop back)
              ▼                         ▼
         ┌──────────────────────────────────┐
         │         Finalize                 │
         │  Return text, score, history     │
         └──────────────┬───────────────────┘
                        │
                        ▼
         ┌──────────────────────────────────┐
         │  Optional: Voice Clone (Fish-TTS)│
         │  Optional: Lip Sync (MuseTalk)   │
         └──────────────┬───────────────────┘
                        │
                        ▼
                 ┌────────────┐
                 │  Output    │
                 │ Video/Text │
                 └────────────┘
```

### Reflection Loop Behavior

| Iteration | Score | Action |
|---|---|---|
| 1 | 0.65 | Retry → re-correct |
| 2 | 0.83 | Retry → re-correct |
| 3 | 0.94 | ✓ Finalize (above threshold) |

If max iterations reached without convergence → returns best attempt with `is_acceptable: false`.

---

## End-to-End Data Flow

```
INPUT
  └─ video.mp4 (speaker with accented English)
              │
              ▼
┌─────────────────────────────────────────────┐
│  Orchestrator: POST /process                │
└──────────────────┬──────────────────────────┘
                   │
                   ├─► extract_audio (ffmpeg) ──► storage/{id}_audio.wav
                   │
                   ├─► transcribe (Whisper) ─────► "He don't like apples..."
                   │
                   ├─► correct (CoEdit) ─────────► "He doesn't like apples..."
                   │       ▲                      │
                   │       │                      ▼
                   │       │                check (CoLA) ──► score: 0.85
                   │       │                                 │
                   │       │     score < 0.9 ◄───────────────┤
                   │       └─────retry (max 3)               │
                   │                                         ▼
                   │                                    score: 0.94 ✓
                   │
                   ▼
           Output: corrected text + score + history

OPTIONAL EXTENSIONS (manual via individual endpoints)
  ├─► translate (NLLB-200)  ─────► Korean/Japanese/etc text
  ├─► tts (Fish-TTS S2-Pro) ─────► cloned voice .wav
  └─► sync (MuseTalk v1.5)  ─────► lip-synced .mp4
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Models** | Whisper-Large-v3-Turbo, CoEdit-Large, BERT-CoLA, NLLB-200, Fish-Speech S2-Pro, MuseTalk v1.5 |
| **Frameworks** | PyTorch 2.1, Transformers 4.44, FastAPI, LangGraph |
| **Agent** | LangGraph StateGraph with conditional reflection edges |
| **Infrastructure** | Docker Compose, NVIDIA Container Toolkit, CUDA 11.8 |
| **Language** | Python 3.10 |
| **API Format** | REST (JSON) for translator/orchestrator, MessagePack for Fish-TTS |
| **Inter-service** | httpx async clients, shared volume mounts |

---

## Prerequisites

### Hardware
- **GPU:** NVIDIA with ≥ 16 GB VRAM (24 GB+ recommended for parallel services)
- **RAM:** 32 GB+ system memory
- **Storage:** 80 GB+ for models and Docker images

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
├── checkpoints/                  ← Model weights (host-only, not committed)
│   ├── whisper/                  Whisper-Large-v3-Turbo
│   ├── grammarly/                CoEdit-Large
│   ├── checker/                  BERT-CoLA (grammar scorer)
│   ├── translator/               NLLB-200-distilled-600M
│   ├── fish_tts/                 Fish-Speech S2-Pro
│   └── musetalk/                 MuseTalk v1.5
│       └── models/
│           ├── musetalk/
│           ├── musetalkV15/
│           ├── sd-vae-ft-mse/
│           ├── whisper/
│           ├── dwpose/
│           └── face-parse-bisent/
│
├── storage/                      ← Shared runtime files
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
│   │   └── Dockerfile            ← Runs Fish-Speech's official server
│   │
│   └── musetalk/
│       ├── Dockerfile
│       ├── api.py
│       └── requirements.txt
│
├── orchestrator/                 ← LangGraph reflection agent
│   ├── Dockerfile
│   ├── main.py                   FastAPI entry point
│   ├── agent.py                  LangGraph workflow
│   ├── state.py                  Pipeline state schema
│   ├── tools.py                  HTTP service clients
│   └── requirements.txt
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

# Fish-TTS S2-Pro (gated — request access at huggingface.co/fishaudio/s2-pro)
huggingface-cli download fishaudio/s2-pro \
    --local-dir checkpoints/fish_tts \
    --include "*.safetensors" "*.json" "*.pth" "*.jinja" "*.model" "*.txt"

# MuseTalk v1.5 (community fork by kevinwang676)
huggingface-cli download kevinwang676/MuseTalk1.5 \
    --local-dir checkpoints/musetalk \
    --include "*.pt" "*.pth" "*.bin" "*.safetensors" "*.json" \
    --max-workers 1
```

**Total download size: ~26 GB**. For long downloads over SSH:
```bash
nohup huggingface-cli download <repo> --local-dir <path> > download.log 2>&1 &
tail -f download.log
```

### 3. Build Docker Containers

```bash
# Build all services
docker compose build

# Or build individually (recommended for first time)
docker compose build orchestrator   # ~2 min
docker compose build translator     # ~15 min
docker compose build fish_tts       # ~25 min
docker compose build musetalk       # ~30 min
```

### 4. Start Services

```bash
# Start orchestrator + translator (typical reflection workflow)
docker compose up -d translator orchestrator

# Add fish_tts for voice cloning
docker compose stop translator
docker compose up -d fish_tts

# Add musetalk for lip-sync
docker compose stop fish_tts
docker compose up -d musetalk

# Verify
docker ps
```

### 5. Health Checks

```bash
curl http://localhost:8000/health    # Orchestrator
curl http://localhost:8005/health    # Translator
curl http://localhost:8003/v1/health 2>&1 | head -5  # Fish-TTS
curl http://localhost:8004/health    # MuseTalk
```

---

## API Reference

### Orchestrator Service (port 8000)

#### `POST /process`
Run full reflection pipeline on uploaded video.

```python
import requests

with open("video.mp4", "rb") as f:
    r = requests.post(
        "http://localhost:8000/process",
        files={"video": f},
        data={"max_iterations": "3"},
        timeout=600
    )

result = r.json()
# {
#   "job_id"          : "...",
#   "transcript"      : "He don't like apples...",
#   "final_text"      : "He doesn't like apples...",
#   "grammar_score"   : 0.94,
#   "is_acceptable"   : true,
#   "iterations_used" : 1,
#   "max_iterations"  : 3,
#   "audio_path"      : "/app/storage/.../audio.wav",
#   "history"         : [...]
# }
```

---

### Translator Service (port 8005)

#### `POST /transcribe`
```python
r = requests.post("http://localhost:8005/transcribe", json={
    "audio_path": "/app/storage/input.mp4"
})
```

#### `POST /correct`
```python
r = requests.post("http://localhost:8005/correct", json={
    "text": "He don't like apples"
})
# Response: {"corrected": "He doesn't like apples."}
```

#### `POST /check`
```python
r = requests.post("http://localhost:8005/check", json={
    "text": "She goes to school every day."
})
# Response: {"grammar_score": 0.94, "is_acceptable": true}
```

#### `POST /translate`
```python
r = requests.post("http://localhost:8005/translate", json={
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "kor_Hang"
})
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
```python
with open("video.mp4", "rb") as f:
    r = requests.post(
        "http://localhost:8005/extract_audio",
        files={"file": f}
    )
```

---

### Fish-TTS Service (port 8003)

Uses Fish-Speech's official MessagePack API.

#### `POST /v1/tts`
```python
import requests
import ormsgpack

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
    "temperature": 0.7,
    "streaming": False
}

r = requests.post(
    "http://localhost:8003/v1/tts",
    headers={"content-type": "application/msgpack"},
    data=ormsgpack.packb(payload),
    timeout=600
)
```

**Reference audio requirements:**
- Format: 22050 Hz mono WAV
- Duration: 5–30 seconds
- Quality: clean speech, minimal background noise
- Reference text MUST match audio exactly

---

### MuseTalk Service (port 8004)

#### `POST /sync`
```python
import requests

with open("video.mp4", "rb") as v, open("audio.wav", "rb") as a:
    r = requests.post(
        "http://localhost:8004/sync",
        files = {"video": v, "audio": a},
        data  = {
            "bbox_shift"  : "0",
            "fps"         : "25",
            "use_float16" : "true"
        },
        timeout = 900
    )
```

**Requirements:**
- Front-facing visible face in video
- Audio length ≥ video length
- Recommended duration: 5–30 seconds
- VRAM: ~10 GB during inference

---

## Full Pipeline Example

```python
import requests
import ormsgpack
import torchaudio
import shutil

INPUT_VIDEO = "/path/to/input.mp4"
TARGET_LANG = "kor_Hang"

# ─── 1. Run reflection pipeline (transcribe → correct → check) ───
with open(INPUT_VIDEO, "rb") as f:
    r = requests.post(
        "http://localhost:8000/process",
        files={"video": f},
        data={"max_iterations": "3"},
        timeout=600
    )
result = r.json()
corrected = result["final_text"]
print(f"Corrected (score {result['grammar_score']:.3f}): {corrected}")

# ─── 2. Translate ───
r = requests.post("http://localhost:8005/translate", json={
    "text": corrected,
    "source_lang": "eng_Latn",
    "target_lang": TARGET_LANG
})
translated = r.json()["translated"]

# ─── 3. Extract reference audio for voice clone ───
audio_path = result["audio_path"]
shutil.copy(audio_path, "storage/reference.wav")

waveform, sr = torchaudio.load("storage/reference.wav")
if sr != 22050:
    waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
torchaudio.save("storage/reference_22k.wav", waveform[:, :22050*20], 22050)

# ─── 4. Voice clone with translated text ───
with open("storage/reference_22k.wav", "rb") as f:
    ref_audio = f.read()

payload = {
    "text": translated,
    "references": [{"audio": ref_audio, "text": result["transcript"]}],
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

# ─── 5. Lip-sync ───
with open(INPUT_VIDEO, "rb") as v, open("storage/cloned_voice.wav", "rb") as a:
    r = requests.post(
        "http://localhost:8004/sync",
        files={"video": v, "audio": a},
        data={"bbox_shift": "0", "fps": "25", "use_float16": "true"},
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
| Orchestrator | 0 GB | <1s | <1s | ~135 MB |
| Translator | ~8 GB | ~30s | ~3-10s | ~12 GB |
| Fish-TTS | ~10 GB | ~60s | ~5-15s | ~16 GB |
| MuseTalk | ~10 GB | ~3 min | ~1-3 min | ~12 GB |

**Running all 3 GPU services simultaneously requires 28+ GB VRAM.** On smaller GPUs (16-24 GB), run sequentially via service swapping:

```bash
docker compose stop fish_tts musetalk
docker compose up -d translator orchestrator
# Run reflection pipeline
# ...
docker compose stop translator
docker compose up -d fish_tts
# Run voice clone
# ...
docker compose stop fish_tts
docker compose up -d musetalk
# Run lip sync
```

---

## Common Issues

### CUDA Out of Memory
```bash
docker compose stop translator
nvidia-smi   # verify freed
docker compose up -d fish_tts
```

### Container Won't Start
```bash
docker compose logs <service_name> --tail 50

# Common causes:
# - Empty Python files (VS Code save issue): ls -lh services/<name>/
# - Missing model checkpoints: ls checkpoints/<name>/
# - Port conflict: ss -tlnp | grep <port>
```

### Model Download Fails
```bash
huggingface-cli logout
huggingface-cli login

# Resume with single thread (avoids race conditions)
huggingface-cli download <repo> --local-dir <path> --max-workers 1
```

### Fish-TTS Decoder Config Error
```
Use --decoder-config-name modded_dac_vq, NOT firefly_gan_vq.
The firefly_gan_vq config is from older Fish-Speech v1.4 docs.
v2.0.0+ uses modded_dac_vq for S2-Pro architecture.
```

### MuseTalk pkg_resources Error
```
Error: ModuleNotFoundError: No module named 'pkg_resources'

Cause: setuptools 80+ removed pkg_resources module.
Fix: Pin to setuptools<80 in Dockerfile.
```

### MuseTalk VAE Path Mismatch
```
Error: Repository Not Found for url: .../models/sd-vae/...

Cause: MuseTalk expects "sd-vae" folder but checkpoints have "sd-vae-ft-mse".
Fix: api.py creates a symlink at startup:
  os.symlink("/opt/MuseTalk/models/sd-vae-ft-mse", "/opt/MuseTalk/models/sd-vae")
```

---

## Development

### Adding a New Service

1. Create `services/<name>/` directory with Dockerfile, api.py, requirements.txt
2. Add service block to `docker-compose.yml`
3. Mount required checkpoint volumes
4. Update orchestrator's `tools.py` with HTTP client for new service
5. Add LangGraph node in `agent.py` if part of pipeline
6. Build & test:
   ```bash
   docker compose build <name>
   docker compose up -d <name>
   ```

### Modifying API Code

```bash
docker compose down <service>
docker compose build <service>     # Code is baked via COPY
docker compose up -d <service>
```

### Updating Checkpoints

Checkpoints are mounted as volumes — no rebuild needed:
```bash
# Replace files in checkpoints/<name>/
docker compose restart <service>
```

### Extending The Reflection Agent

The reflection loop is defined in `orchestrator/agent.py`. Add new quality checks:

```python
async def node_check_semantic(state):
    """Check if corrected text preserves original meaning."""
    similarity = await tools.compute_similarity(
        state["transcript"],
        state["corrected"]
    )
    return {**state, "semantic_score": similarity}

# Add to graph
workflow.add_node("check_semantic", node_check_semantic)
workflow.add_edge("check", "check_semantic")
workflow.add_conditional_edges(
    "check_semantic",
    lambda s: "retry" if s["semantic_score"] < 0.85 else "finalize",
    {"retry": "correct", "finalize": "finalize"}
)
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
- MuseTalk (MIT, via TMElyralab)

**Note:** Several models are non-commercial. For commercial deployment, replace with appropriate alternatives.

---

## Acknowledgments

- OpenAI for Whisper
- Grammarly for CoEdit
- Meta AI for NLLB-200
- Fish Audio for Fish-Speech S2-Pro
- TMElyralab for MuseTalk
- HuggingFace for the model hub
- LangChain team for LangGraph

---

## Citation

If you use this work in research:

```bibtex
@software{accentflow2026,
  author  = {Md Rezanur Islam, Kangbin Yim},
  title   = {AccentFlow: Agentic AI Pipeline for Accent-Adaptive Video Synthesis with Reflection-Based Quality Control},
  year    = {2026},
  url     = {https://github.com/Arupreza/AccentFlow}
}
```