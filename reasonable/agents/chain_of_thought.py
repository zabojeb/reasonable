import json, time
from dataclasses import dataclass
from typing import List, Callable, Optional

from ..llms.base import BaseLLM
from .. import prompts as _p

__all__ = ["ChainOfThoughtAgent", "ReasoningResult"]

@dataclass
class ReasoningResult:
    thoughts: List[str]
    answer: str

class ChainOfThoughtAgent:
    """Iterative reasoning agent using user‑defined prompt templates."""

    def __init__(
        self,
        llm: BaseLLM,
        thoughts_prompt: Callable[[str, List[str]], str] = _p.thoughts_prompt,
        answer_prompt: Callable[[str, List[str]], str] = _p.answer_prompt,
        max_steps: int = 8,
        sleep: float = 0,
        verbose: bool = False,
    ):
        self.llm = llm
        self.thoughts_prompt = thoughts_prompt
        self.answer_prompt = answer_prompt
        self.max_steps = max_steps
        self.sleep = sleep
        self.verbose = verbose

    def reason(self, user_query: str) -> ReasoningResult:
        thoughts: List[str] = []
        for step in range(self.max_steps):
            prompt = self.thoughts_prompt(user_query, thoughts)
            if self.verbose:
                print(f"\n[step {step}] prompt:\n{prompt}\n")
            response = self.llm.generate(prompt)
            if self.verbose:
                print(f"[step {step}] LLM output: {response}")

            thought = self._extract_tag(response, "reasoning") or response.strip()
            next_action = self._extract_tag(response, "next_action") or "stop"
            thoughts.append(thought)
            if next_action.lower().strip() != "continue":
                break
            if self.sleep:
                time.sleep(self.sleep)

        final_prompt = self.answer_prompt(user_query, thoughts)
        answer = self.llm.generate(final_prompt).strip()
        return ReasoningResult(thoughts, answer)

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        import re
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.S)
        return m.group(1).strip() if m else ""
