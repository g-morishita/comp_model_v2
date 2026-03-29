"""Inference-specific exception types."""


class SamplingError(RuntimeError):
    """Raised when a backend sampling procedure fails.

    Parameters
    ----------
    message
        Human-readable description of the failure.
    original
        The original exception raised by the backend, if any.
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original
