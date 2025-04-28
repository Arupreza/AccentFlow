## 📖 Project Overview

In this project, users upload a video file, and the app first extracts and transcribes the audio using OpenAI’s Whisper model. The raw transcription is then corrected using a T5-based grammar correction model to improve grammar, spelling, and fluency, while also highlighting the changes compared to the original text. A 5-second audio clip from the uploaded video is trimmed to serve as a voice reference. Using the XTTS-v2 model from Coqui, the app performs voice cloning to synthesize the corrected text into natural-sounding speech that matches the original speaker’s voice. Finally, the generated voice audio is made available for playback directly within the app, completing an end-to-end process of transcription, grammar enhancement, and speaker-adapted text-to-speech synthesis.

---

## 📂 Project Structure

. ├── Assets_Text.py ├── requirements.txt ├── Streamlit_App.py ├── saved_output/ │ ├── extracted_audio_segments/ │ ├── transcribed_subtitles.srt │ ├── incorrect_paragraph.txt │ ├── corrected_paragraph.txt │ ├── full_audio.wav │ ├── trimmed_ref.wav │ └── output.wav └── .env
