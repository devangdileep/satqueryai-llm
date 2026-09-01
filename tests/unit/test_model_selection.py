import pytest
from app.agent.model_selector import model_selector
from app.schemas.analysis import QueryAnalysis
from app.schemas.images import ImageInputConfig, ImageMetadata


@pytest.mark.asyncio
async def test_capability_based_model_selection_geochat():
    analysis = QueryAnalysis(
        intent="scene_understanding",
        task="single_image_vqa",
        raw_query="What is shown in this image?"
    )
    input_config = ImageInputConfig(
        configuration_type="single_image",
        image_count=1,
        images=[
            ImageMetadata(
                filename="optical_test.png",
                width=504,
                height=504,
                num_bands=3,
                dtype="uint8",
                modality="optical"
            )
        ]
    )

    selection = await model_selector.select_model(analysis, input_config)
    assert selection.selected_model.lower() == "geochat"


@pytest.mark.asyncio
async def test_capability_based_model_selection_changechat():
    analysis = QueryAnalysis(
        intent="change_analysis",
        task="multitemporal_change_vqa",
        requires_temporal_reasoning=True,
        raw_query="What changed?"
    )
    input_config = ImageInputConfig(
        configuration_type="bitemporal_pair",
        image_count=2,
        images=[
            ImageMetadata(filename="t1.png", width=256, height=256, num_bands=3, dtype="uint8", modality="optical"),
            ImageMetadata(filename="t2.png", width=256, height=256, num_bands=3, dtype="uint8", modality="optical")
        ]
    )

    selection = await model_selector.select_model(analysis, input_config)
    assert selection.selected_model.lower() == "changechat"


@pytest.mark.asyncio
async def test_capability_based_model_selection_sar_fusion():
    analysis = QueryAnalysis(
        intent="multimodal_fusion",
        task="optical_sar_analysis",
        requires_multimodal_reasoning=True,
        raw_query="Analyze optical and SAR together."
    )
    input_config = ImageInputConfig(
        configuration_type="optical_sar_pair",
        image_count=2,
        images=[
            ImageMetadata(filename="opt.png", width=256, height=256, num_bands=3, dtype="uint8", modality="optical"),
            ImageMetadata(filename="sar.tif", width=256, height=256, num_bands=2, dtype="float32", modality="sar")
        ]
    )

    selection = await model_selector.select_model(analysis, input_config)
    assert selection.selected_model.lower() == "sar-ml-fusion"
