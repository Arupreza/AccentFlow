from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from agent import graph
import shutil, uuid, os

app = FastAPI(title="AccentFlow Reflection Agent")

STORAGE_DIR = "/app/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "orchestrator"}


@app.post("/process")
async def process(
    video: UploadFile = File(...),
    max_iterations: int = Form(3)
):
    """
    Reflection agent pipeline:
        1. Extract audio from uploaded video
        2. Transcribe audio
        3. Correct grammar
        4. Check grammar score
        5. If score < 0.9 → retry correction (up to max_iterations)
        6. Return final corrected text + full history

    Returns:
        JSON with transcript, final corrected text, score, and reflection history
    """
    job_id = str(uuid.uuid4())
    video_path = os.path.join(STORAGE_DIR, f"{job_id}.mp4")

    # Save upload
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    initial_state = {
        "video_path": video_path,
        "audio_path": None,
        "transcript": None,
        "corrected": None,
        "grammar_score": None,
        "is_acceptable": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_text": None,
        "history": [],
        "error": None
    }

    try:
        final_state = await graph.ainvoke(initial_state)

        if final_state.get("error"):
            return JSONResponse({
                "error": final_state["error"],
                "history": final_state.get("history", [])
            }, status_code=500)

        return {
            "job_id"          : job_id,
            "transcript"      : final_state["transcript"],
            "final_text"      : final_state["final_text"],
            "grammar_score"   : final_state["grammar_score"],
            "is_acceptable"   : final_state["is_acceptable"],
            "iterations_used" : final_state["iteration"],
            "max_iterations"  : final_state["max_iterations"],
            "audio_path"      : final_state["audio_path"],
            "history"         : final_state["history"]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)