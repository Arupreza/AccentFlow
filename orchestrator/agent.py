from langgraph.graph import StateGraph, END
from state import ReflectionState
import tools

THRESHOLD = 0.9


# ───────── Node functions ─────────

async def node_extract_audio(state: ReflectionState) -> ReflectionState:
    try:
        audio_path = await tools.extract_audio(state["video_path"])
        return {**state, "audio_path": audio_path}
    except Exception as e:
        return {**state, "error": f"extract_audio failed: {e}"}


async def node_transcribe(state: ReflectionState) -> ReflectionState:
    if state.get("error"):
        return state
    try:
        transcript = await tools.transcribe(state["audio_path"])
        history = state.get("history", []) + [{
            "step": "transcribe",
            "output": transcript
        }]
        return {**state, "transcript": transcript, "history": history}
    except Exception as e:
        return {**state, "error": f"transcribe failed: {e}"}


async def node_correct(state: ReflectionState) -> ReflectionState:
    if state.get("error"):
        return state
    try:
        # On first iteration, correct the transcript
        # On retries, correct the previously corrected text further
        input_text = state.get("corrected") or state["transcript"]
        corrected = await tools.correct_grammar(input_text)

        history = state.get("history", []) + [{
            "step": f"correct (iter {state.get('iteration', 0)})",
            "input": input_text,
            "output": corrected
        }]
        return {
            **state,
            "corrected": corrected,
            "iteration": state.get("iteration", 0) + 1,
            "history": history
        }
    except Exception as e:
        return {**state, "error": f"correct failed: {e}"}


async def node_check(state: ReflectionState) -> ReflectionState:
    if state.get("error"):
        return state
    try:
        result = await tools.check_grammar(state["corrected"])
        score = result["grammar_score"]
        is_acceptable = score >= THRESHOLD

        history = state.get("history", []) + [{
            "step": f"check (iter {state['iteration']})",
            "score": score,
            "acceptable": is_acceptable
        }]

        return {
            **state,
            "grammar_score": score,
            "is_acceptable": is_acceptable,
            "history": history
        }
    except Exception as e:
        return {**state, "error": f"check failed: {e}"}


async def node_finalize(state: ReflectionState) -> ReflectionState:
    if state.get("error"):
        return state
    return {**state, "final_text": state["corrected"]}


# ───────── Conditional edge ─────────

def should_retry(state: ReflectionState) -> str:
    """
    Route based on grammar quality.
    - score >= 0.9 → finalize
    - score < 0.9 AND iteration < max → retry correction
    - score < 0.9 AND iteration >= max → finalize anyway (best effort)
    """
    if state.get("error"):
        return "finalize"

    if state.get("is_acceptable"):
        return "finalize"

    if state["iteration"] >= state["max_iterations"]:
        return "finalize"

    return "retry"


# ───────── Build graph ─────────

def build_graph():
    workflow = StateGraph(ReflectionState)

    workflow.add_node("extract_audio", node_extract_audio)
    workflow.add_node("transcribe",    node_transcribe)
    workflow.add_node("correct",       node_correct)
    workflow.add_node("check",         node_check)
    workflow.add_node("finalize",      node_finalize)

    workflow.set_entry_point("extract_audio")
    workflow.add_edge("extract_audio", "transcribe")
    workflow.add_edge("transcribe",    "correct")
    workflow.add_edge("correct",       "check")

    # Conditional reflection loop
    workflow.add_conditional_edges(
        "check",
        should_retry,
        {
            "retry"    : "correct",   # loop back
            "finalize" : "finalize"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


graph = build_graph()