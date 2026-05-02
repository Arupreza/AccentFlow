FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONPATH=/opt/MuseTalk

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libsm6 libxext6 libgl1 libglib2.0-0 \
    libsox-dev sox tzdata wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-*-amd64-static* && \
    ffmpeg -version | head -1

RUN git clone https://github.com/TMElyralab/MuseTalk.git /opt/MuseTalk

WORKDIR /opt/MuseTalk
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=68,<80" wheel && \
    pip install --no-cache-dir -r requirements.txt || echo "Some deps failed, continuing"

RUN pip install --no-cache-dir mmengine==0.10.4 && \
    pip install --no-cache-dir "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html && \
    pip install --no-cache-dir "mmdet==3.2.0"

RUN pip install --no-cache-dir mmpose==1.3.1 --no-deps && \
    pip install --no-cache-dir json-tricks munkres xtcocotools

# CRITICAL: Force setuptools <80 (must come AFTER all mmlab installs)
# pkg_resources module was removed in setuptools 80+
RUN pip install --no-cache-dir --force-reinstall "setuptools>=68,<80"

# Verify pkg_resources works
RUN python -c "import pkg_resources; print('pkg_resources OK:', pkg_resources.__file__)"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8004"]
