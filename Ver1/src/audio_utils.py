import os, json, shutil, subprocess
import streamlit as st
from pydub import AudioSegment
from .ui import add_log

# ---------- extraction & trims ----------
def extract_audio_segments(video_path: str, out_dir: str, seg_seconds: float):
    try:
        if not os.path.exists(video_path):
            add_log(f"Video not found: {video_path}", "error"); return None
        os.makedirs(out_dir, exist_ok=True)
        full = AudioSegment.from_file(video_path)
        full_wav = os.path.join(out_dir, "full_audio.wav")
        full.export(full_wav, format="wav")
        st.session_state.generated_files["full_audio"] = full_wav

        seg_ms = max(1000, int(seg_seconds * 1000))
        n = max(1, len(full) // seg_ms)
        add_log(f"Creating {n} segment(s)", "info")
        for i in range(n):
            part = full[i*seg_ms : min((i+1) * seg_ms, len(full))]
            p = os.path.join(out_dir, f"segment_{i+1}.wav")
            part.export(p, format="wav")
            st.session_state.generated_files[f"segment_{i+1}"] = p
        add_log("✅ Audio extraction complete", "success")
        return full_wav
    except Exception as e:
        add_log(f"❌ Extract audio error: {e}", "error"); return None

def trim_reference(inp: str, out: str, seconds=5) -> bool:
    try:
        if not os.path.exists(inp):
            add_log(f"Ref audio not found: {inp}", "error"); return False
        a = AudioSegment.from_file(inp)
        (a if len(a) <= seconds*1000 else a[:seconds*1000]).export(out, format="wav")
        st.session_state.generated_files["trimmed_ref"] = out
        return True
    except Exception as e:
        add_log(f"❌ Trim error: {e}", "error"); return False

# ---------- ffmpeg helpers ----------
def _which_ffmpeg(): return shutil.which("ffmpeg")

def ffprobe_duration_seconds(path: str):
    if not _which_ffmpeg() or not os.path.isfile(path): return None
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","json", path],
            capture_output=True, text=True
        )
        if r.returncode != 0: return None
        data = json.loads(r.stdout or "{}")
        d = data.get("format", {}).get("duration")
        return float(d) if d is not None else None
    except Exception:
        return None

def merge_audio_with_video(video_in: str, audio_in: str, out_path: str) -> bool:
    try:
        if not os.path.isfile(video_in): add_log(f"Video not found: {video_in}", "error"); return False
        if not os.path.isfile(audio_in): add_log(f"Audio not found: {audio_in}", "error"); return False
        if not _which_ffmpeg(): add_log("ffmpeg not found.", "error"); return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cmd = [
            "ffmpeg","-y","-i",video_in,"-i",audio_in,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","copy","-c:a","aac","-b:a","192k","-ar","48000","-ac","2",
            "-shortest","-movflags","+faststart","-pix_fmt","yuv420p", out_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(out_path):
            if r.stderr: add_log(r.stderr[:2000], "error")
            if r.stdout: add_log(r.stdout[:2000], "warning")
            return False
        st.session_state.generated_files["merged_video"] = out_path
        add_log("✅ Merged audio into video", "success")
        return True
    except Exception as e:
        add_log(f"❌ Merge error: {e}", "error"); return False

# ---------- time-stretch ----------
def _split_atempo_chain(factor: float):
    if factor <= 0: factor = 1.0
    chain = []
    while factor > 2.0 + 1e-6: chain.append(2.0); factor /= 2.0
    while factor < 0.5 - 1e-6: chain.append(0.5); factor /= 0.5
    chain.append(factor)
    return [max(0.5, min(2.0, f)) for f in chain]

def time_stretch_wav_ffmpeg(in_wav: str, factor: float, out_wav: str) -> bool:
    try:
        if not _which_ffmpeg(): add_log("ffmpeg not found; cannot stretch.", "error"); return False
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        filt = ",".join([f"atempo={f:.6f}" for f in _split_atempo_chain(float(factor))])
        r = subprocess.run(["ffmpeg","-y","-i",in_wav,"-af",filt,"-c:a","pcm_s16le",out_wav],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(out_wav):
            if r.stderr: add_log(r.stderr[:2000], "error")
            if r.stdout: add_log(r.stdout[:2000], "warning")
            return False
        return True
    except Exception as e:
        add_log(f"❌ Stretch error: {e}", "error"); return False