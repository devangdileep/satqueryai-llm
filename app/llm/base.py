from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract provider interface for LLM operations."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        temperature: float = 0.0
    ) -> T:
        """Generate structured response validated against a Pydantic schema."""
        pass
