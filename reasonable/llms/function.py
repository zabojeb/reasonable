from typing import Callable, Any
from .base import BaseLLM

class FunctionLLM(BaseLLM):
    """Wraps a callable returning text into a BaseLLM."""
    def __init__(self, func: Callable[[str], str]):
        self.func = func

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.func(prompt, **kwargs)
