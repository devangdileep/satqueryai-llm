import json
from typing import Type, TypeVar
from pydantic import BaseModel
from groq import AsyncGroq
from app.core.config import settings
from app.core.logging import logger
from app.llm.base import LLMProvider

T = TypeVar("T", bound=BaseModel)


class GroqProvider(LLMProvider):
    """Groq API provider with structured JSON output and Pydantic validation."""

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.model = settings.GROQ_MODEL
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: Type[T],
        temperature: float = 0.0
    ) -> T:
        if not self.client:
            raise ValueError("GROQ_API_KEY is not configured.")

        json_schema_prompt = (
            f"{system_prompt}\n\n"
            f"You MUST respond ONLY with a single valid JSON object matching this JSON Schema:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}\n"
            f"Do NOT include markdown formatting, code blocks, or explanatory text before or after the JSON."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": json_schema_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            parsed_json = json.loads(content)
            return response_model.model_validate(parsed_json)
        except Exception as e:
            logger.error("groq_generation_failed", error=str(e))
            raise RuntimeError(f"Groq structured generation failed: {str(e)}")
