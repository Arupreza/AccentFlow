FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /opt/fish-speech

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential libsox-dev sox portaudio19-dev tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[stable]"

CMD ["python", "tools/api_server.py", \
    "--listen", "0.0.0.0:8003", \
    "--llama-checkpoint-path", "/app/checkpoints", \
    "--decoder-checkpoint-path", "/app/checkpoints/codec.pth", \
    "--decoder-config-name", "modded_dac_vq", \
    "--half", \
    "--compile"]