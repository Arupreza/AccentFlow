FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ── Block all HuggingFace network calls — model is mounted from host ────────
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    which uvicorn

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8005"]