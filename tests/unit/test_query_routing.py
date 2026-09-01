import pytest
from app.agent.query_analyzer import QueryAnalyzer
from app.llm.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_query_routing_change():
    analyzer = QueryAnalyzer(provider=MockLLMProvider())
    analysis = await analyzer.analyze("What changed around the settlement between these two images?")

    assert analysis.intent == "change_analysis"
    assert analysis.task == "multitemporal_change_vqa"
    assert analysis.requires_temporal_reasoning is True


@pytest.mark.asyncio
async def test_query_routing_grounding():
    analyzer = QueryAnalyzer(provider=MockLLMProvider())
    analysis = await analyzer.analyze("Where are the water bodies in this image?")

    assert analysis.intent == "object_localization"
    assert analysis.task == "region_grounding"
    assert analysis.requires_spatial_evidence is True


@pytest.mark.asyncio
async def test_query_routing_sar_fusion():
    analyzer = QueryAnalyzer(provider=MockLLMProvider())
    analysis = await analyzer.analyze("Use the optical and SAR images together to identify built-up regions.")

    assert analysis.intent == "multimodal_fusion"
    assert analysis.task == "optical_sar_analysis"
    assert analysis.requires_multimodal_reasoning is True
