from app.core.config import settings
from app.core.logging import logger
from app.llm.base import LLMProvider
from app.llm.groq import GroqProvider
from app.llm.mock import MockLLMProvider
from app.schemas.analysis import QueryAnalysis


class QueryAnalyzer:
    """Converts natural language remote-sensing questions into structured QueryAnalysis task specs."""

    def __init__(self, provider: LLMProvider | None = None):
        if provider:
            self.provider = provider
        elif settings.LLM_PROVIDER == "groq" and settings.GROQ_API_KEY:
            self.provider = GroqProvider()
        else:
            self.provider = MockLLMProvider()

    async def analyze(self, query: str) -> QueryAnalysis:
        system_prompt = (
            "You are an expert remote-sensing query analysis agent. "
            "Analyze the natural language question and extract: intent, task, target_entities, "
            "requested_outputs, and required reasoning capabilities (spatial, temporal, multimodal)."
        )
        try:
            analysis = await self.provider.generate_structured(
                prompt=query,
                system_prompt=system_prompt,
                response_model=QueryAnalysis
            )
            logger.info("query_analyzed", query=query, intent=analysis.intent, task=analysis.task)
            return analysis
        except Exception as e:
            logger.error("query_analysis_failed", error=str(e))
            # Fallback
            return QueryAnalysis(
                intent="scene_understanding",
                task="single_image_vqa",
                raw_query=query
            )
