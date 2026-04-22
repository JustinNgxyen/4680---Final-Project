from dataclasses import dataclass

@dataclass
class ErrorInput:
    """What the student provides."""
    error_message: str
    code_snippet: str
    language: str = "Python"

    def validate(self) -> None:
        if not self.error_message.strip():
            raise ValueError("error_message cannot be empty.")
        lines = self.code_snippet.strip().splitlines()
        if len(lines) > 30:
            raise ValueError(
                f"code_snippet must be 30 lines or fewer (got {len(lines)}). "
                "Trim the snippet to the section closest to the error."
            )


@dataclass
class ErrorExplanation:
    """What the agent returns to the student."""
    plain_explanation: str   # jargon-free, 1-2 sentences
    likely_cause: str        # single most probable root cause
    debug_steps: list[str]   # 1-3 concrete fix steps
    error_type: str          # e.g. "TypeError", "IndexError"