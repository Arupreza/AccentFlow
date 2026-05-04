from typing import TypedDict, Optional

class ReflectionState(TypedDict):
    # Input
    video_path: str

    # Pipeline outputs
    audio_path: Optional[str]
    transcript: Optional[str]
    corrected: Optional[str]
    grammar_score: Optional[float]
    is_acceptable: Optional[bool]

    # Reflection control
    iteration: int
    max_iterations: int
    final_text: Optional[str]
    history: list

    # Error handling
    error: Optional[str]