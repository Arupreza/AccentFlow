# src/pipeline.py
import os, re, tempfile, traceback
import nltk, torch, streamlit as st
from pydub import AudioSegment

from .config import (
    TTS_MAX_CHARS, SILENCE_PAD_MS,
    LANG_TO_XTTS, DEFAULT_TTS_LANG
)
from .ui import add_log
from .audio_utils import (
    extract_audio_segments, trim_reference, merge_audio_with_video,
    time_stretch_wav_ffmpeg, ffprobe_duration_seconds
)
from .asr import transcribe_video_in_chunks, detect_language_from_text
from .text_utils import format_srt, english_grammar_correction, highlight_corrections
from .models import load_tts_model
from .translate import translate_text

def _sanitize(txt: str) -> str:
    txt = txt.replace("\u200b","").replace("\ufeff","")
    return re.sub(r"[^\S\r\n]+", " ", txt).strip()

def _chunk_for_tts(text: str, max_chars: int):
    sents = nltk.sent_tokenize(text)
    chunks, cur = [], ""
    for s in sents:
        s = s.strip()
        if not s: continue
        if len(cur) + len(s) + 1 <= max_chars: cur = (cur + " " + s).strip()
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks

def _split_text_by_durations(text: str, durations):
    words = _sanitize(text).split()
    if not durations: return [" ".join(words)]
    total_d = sum(max(d,0) for d in durations) or 1.0
    total_w = len(words)
    if total_w == 0: return [""]*len(durations)
    targets = [max(0, round(total_w*(d/total_d))) for d in durations]
    drift, i = total_w - sum(targets), 0
    while drift != 0 and targets:
        if drift > 0: targets[i%len(targets)] += 1; drift -= 1
        else:
            if targets[i%len(targets)] > 0: targets[i%len(targets)] -= 1; drift += 1
        i += 1
    out, idx = [], 0
    for n in targets:
        out.append(" ".join(words[idx:idx+n])); idx += n
    if idx < total_w: out[-1] = (out[-1] + " " + " ".join(words[idx:])).strip()
    return out

def _tts_one(tts, text: str, ref_wav: str, tts_lang: str) -> str:
    if not text.strip():
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
        AudioSegment.silent(duration=1).export(tmp.name, format="wav")
        return tmp.name
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False); tmp.close()
    tts.tts_to_file(text=text, speaker_wav=[ref_wav], language=tts_lang, file_path=tmp.name)
    return tmp.name

def _build_aligned_track(tts, final_text, ref_wav, whisper_chunks, out_path, video_sec, tts_lang):
    try:
        starts, durs = [], []
        for ch in whisper_chunks:
            s,e = ch.get("timestamp", [None,None])
            if s is None or e is None or e <= s: continue
            starts.append(float(s)); durs.append(float(e)-float(s))
        if not starts:
            add_log("No valid ASR timestamps; falling back to global TTS.", "warning")
            return False

        parts = _split_text_by_durations(final_text, durs)
        n = min(len(parts), len(durs)); parts, starts, durs = parts[:n], starts[:n], durs[:n]

        stretched_paths = []
        for i, part in enumerate(parts):
            wav = _tts_one(tts, part, ref_wav, tts_lang)
            cur = len(AudioSegment.from_file(wav))/1000.0
            tgt = max(0.01, durs[i]); factor = tgt/max(0.01, cur)
            outi = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            if not time_stretch_wav_ffmpeg(wav, factor, outi):
                outi = wav
            stretched_paths.append(outi)

        tl = AudioSegment.silent(duration=0)
        cursor = 0
        for i, wav in enumerate(stretched_paths):
            seg = AudioSegment.from_file(wav)
            start_ms = int(starts[i]*1000)
            if start_ms > cursor:
                tl += AudioSegment.silent(duration=(start_ms - cursor))
            tl += seg + AudioSegment.silent(duration=SILENCE_PAD_MS)
            cursor = len(tl)

        if video_sec and video_sec > 0:
            tgt_ms = int(video_sec*1000)
            if len(tl) > tgt_ms: tl = tl[:tgt_ms]
            elif len(tl) < tgt_ms: tl += AudioSegment.silent(duration=tgt_ms-len(tl))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tl.export(out_path, format="wav")
        return True
    except Exception as e:
        add_log(f"Build aligned track error: {e}", "error")
        add_log(traceback.format_exc(), "error")
        return False

def generate_tts_aligned(final_text, ref_path, out_path, tts, whisper_chunks, video_path, tts_lang: str):
    try:
        if not os.path.exists(ref_path):
            add_log(f"Reference not found: {ref_path}", "error"); return False
        if tts is None:
            add_log("TTS not loaded", "error"); return False

        trimmed = os.path.join(os.path.dirname(ref_path), "trimmed_ref.wav")
        if not trim_reference(ref_path, trimmed, seconds=5): return False

        vdur = ffprobe_duration_seconds(video_path)
        if _build_aligned_track(tts, _sanitize(final_text), trimmed, whisper_chunks, out_path, vdur, tts_lang):
            st.session_state.generated_files["tts_output"] = out_path
            add_log("✅ Aligned TTS complete", "success")
            return True

        # fallback: global TTS then global stretch
        parts = _chunk_for_tts(_sanitize(final_text), TTS_MAX_CHARS)
        tmp_parts = []
        for i, p in enumerate(parts, 1):
            t = tempfile.NamedTemporaryFile(suffix=f".p{i}.wav", delete=False); t.close()
            tts.tts_to_file(text=p, speaker_wav=[trimmed], language=tts_lang, file_path=t.name)
            tmp_parts.append(t.name)
        merged = AudioSegment.silent(duration=0)
        for p in tmp_parts: merged += AudioSegment.from_file(p)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.export(out_path, format="wav")

        if vdur and vdur > 0:
            stretched = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            cur = len(AudioSegment.from_file(out_path))/1000.0
            factor = max(0.01, vdur) / max(0.01, cur)
            if time_stretch_wav_ffmpeg(out_path, factor, stretched):
                os.replace(stretched, out_path)

        st.session_state.generated_files["tts_output"] = out_path
        add_log("✅ Global TTS complete", "success")
        return True

    except RuntimeError as e:
        msg = str(e)
        if any(k in msg for k in ("CUDA", "device-side", "out of memory")):
            add_log("CUDA error in TTS → retry on CPU", "warning")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            prev = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                cpu_tts = load_tts_model.__wrapped__()  # reload without cache, on CPU
            finally:
                if prev is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev
            if cpu_tts is None:
                add_log("CPU TTS load failed", "error")
                return False
            return generate_tts_aligned(final_text, ref_path, out_path, cpu_tts, whisper_chunks, video_path, tts_lang)
        add_log(f"TTS error: {e}", "error")
        add_log(traceback.format_exc(), "error")
        return False
    except Exception as e:
        add_log(f"TTS error: {e}", "error")
        add_log(traceback.format_exc(), "error")
        return False

def process_video(
    video_path, output_dir, segment_duration, grammar_model, tts_model,
    source_lang_override: str | None,
    target_lang_choice: str,                    # "same"|"en"|"ko"|"id"
    translate_before_tts: bool
):
    try:
        if not os.path.exists(video_path):
            add_log(f"Video not found: {video_path}", "error"); return False

        os.makedirs(output_dir, exist_ok=True)
        prog = st.progress(0); status = st.empty()
        st.session_state.generated_files = {}

        # 1) Extract reference audio
        status.text("Extracting audio…")
        ref = extract_audio_segments(video_path, output_dir, segment_duration)
        if not ref: return False
        prog.progress(15)

        # 2) ASR (with optional language hint)
        status.text("Transcribing…")
        full_text, chunks = transcribe_video_in_chunks(
            video_path, segment_duration,
            force_lang=source_lang_override
        )
        if not chunks: return False
        prog.progress(40)

        # 3) Save SRT
        status.text("Saving subtitles…")
        srt_path = os.path.join(output_dir, "subtitles.srt")
        open(srt_path, "w", encoding="utf-8").write(format_srt(chunks))
        st.session_state.generated_files["subtitles"] = srt_path
        prog.progress(50)

        # 4) Decide languages
        detected_src = source_lang_override or detect_language_from_text(full_text) or "en"
        tgt_lang = detected_src if target_lang_choice == "same" else target_lang_choice
        xtts_lang = LANG_TO_XTTS.get(tgt_lang, DEFAULT_TTS_LANG)
        add_log(f"Detected source: {detected_src} → Target TTS: {tgt_lang} (XTTS={xtts_lang})", "info")

        # 5) Final text (translation + optional English GEC)
        original_text = full_text
        final_text = original_text

        if translate_before_tts and (detected_src != tgt_lang):
            status.text("Translating transcript…")
            final_text = translate_text(original_text, detected_src, tgt_lang)

        if xtts_lang == "en":
            status.text("Grammar correction (English)…")
            final_text = english_grammar_correction(final_text, grammar_model)
        else:
            status.text("Skipping grammar correction for non-English…")

        wrong = os.path.join(output_dir, "original_transcript.txt")
        corr  = os.path.join(output_dir, "final_text.txt")
        open(wrong, "w", encoding="utf-8").write(original_text)
        open(corr,  "w", encoding="utf-8").write(final_text)
        st.session_state.generated_files["incorrect_text"] = wrong
        st.session_state.generated_files["corrected_text"] = corr
        prog.progress(70)

        # 6) TTS (aligned)
        status.text("Aligned TTS…")
        out_wav = os.path.join(output_dir, "output_aligned.wav")
        if not generate_tts_aligned(final_text, ref, out_wav, tts_model, chunks, video_path, xtts_lang):
            return False
        st.session_state.generated_files["final_audio"] = out_wav
        prog.progress(90)

        # 7) Merge into original video
        status.text("Merging audio + video…")
        merged = os.path.join(output_dir, "merged_video.mp4")
        if not merge_audio_with_video(video_path, out_wav, merged):
            return False
        st.session_state.generated_files["merged_video"] = merged

        prog.progress(100); status.text("Complete ✅")
        st.session_state.processed = True
        st.session_state.incorrect_text = original_text
        st.session_state.corrected_text = final_text
        st.session_state.highlighted_diff = (
            highlight_corrections(original_text, final_text) if xtts_lang == "en" else final_text
        )
        add_log("✅ All steps finished", "success")
        return True
    except Exception as e:
        add_log(f"Pipeline error: {e}", "error")
        add_log(traceback.format_exc(), "error")
        return False