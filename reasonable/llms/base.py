from abc import ABC, abstractmethod
from typing import Any

class BaseLLM(ABC):
    """Abstract interface for any text-generation model."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError
