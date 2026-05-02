import os

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/local/bin/ffmpeg"
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import shutil, uuid, subprocess

app = FastAPI()

STORAGE_DIR  = "/app/storage"
MUSETALK_DIR = "/opt/MuseTalk"

# Symlink mounted checkpoints into MuseTalk's expected location
EXPECTED_CKPT = "/opt/MuseTalk/models"
ACTUAL_CKPT   = "/app/checkpoints/models"

if not os.path.exists(EXPECTED_CKPT):
    os.symlink(ACTUAL_CKPT, EXPECTED_CKPT)

# VAE folder name mismatch fix
VAE_EXPECTED = "/opt/MuseTalk/models/sd-vae"
VAE_ACTUAL   = "/opt/MuseTalk/models/sd-vae-ft-mse"
if os.path.exists(VAE_ACTUAL) and not os.path.exists(VAE_EXPECTED):
    os.symlink(VAE_ACTUAL, VAE_EXPECTED)

os.makedirs(STORAGE_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sync")
async def sync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    bbox_shift: int   = Form(0),
    fps: int          = Form(25),
    use_float16: bool = Form(True)
):
    job_id      = str(uuid.uuid4())
    video_path  = os.path.join(STORAGE_DIR, f"{job_id}_video.mp4")
    audio_path  = os.path.join(STORAGE_DIR, f"{job_id}_audio.wav")
    output_dir  = os.path.join(STORAGE_DIR, f"{job_id}_output")
    os.makedirs(output_dir, exist_ok=True)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    config_path = os.path.join(STORAGE_DIR, f"{job_id}_config.yaml")
    config_content = f"""task_0:
 video_path: "{video_path}"
 audio_path: "{audio_path}"
 bbox_shift: {bbox_shift}
"""
    with open(config_path, "w") as f:
        f.write(config_content)

    cmd = [
        "python", "-m", "scripts.inference",
        "--ffmpeg_path", "/usr/local/bin",
        "--vae_type", "sd-vae",
        "--inference_config", config_path,
        "--result_dir", output_dir,
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--whisper_dir", "models/whisper",
        "--version", "v15",
        "--fps", str(fps),
        "--bbox_shift", str(bbox_shift)
    ]

    if use_float16:
        cmd.append("--use_float16")

    env = os.environ.copy()
    env["PATH"] = "/usr/local/bin:" + env.get("PATH", "")

    try:
        result = subprocess.run(
            cmd,
            cwd=MUSETALK_DIR,
            capture_output=True,
            text=True,
            timeout=900,
            env=env
        )

        if result.returncode != 0:
            return JSONResponse({
                "error"  : "MuseTalk inference failed",
                "stderr" : result.stderr[-3000:],
                "stdout" : result.stdout[-1500:]
            }, status_code=500)

        # Find any .mp4 in output_dir
        mp4_files = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".mp4"):
                    mp4_files.append(os.path.join(root, f))

        if not mp4_files:
            return JSONResponse({
                "error" : "Output video not produced",
                "logs"  : result.stdout[-1500:],
                "files" : os.listdir(output_dir)
            }, status_code=500)

        output_path = max(mp4_files, key=os.path.getsize)

        return FileResponse(
            output_path,
            media_type = "video/mp4",
            filename   = f"synced_{job_id}.mp4"
        )

    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Inference timeout (>15min)"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
