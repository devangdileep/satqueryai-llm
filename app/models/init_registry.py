from app.models.bigearthnet_vlm.adapter import BigEarthNetVLMAdapter
from app.models.changechat.adapter import ChangeChatAdapter
from app.models.geochat.adapter import GeoChatAdapter
from app.models.prithvi.adapter import PrithviAdapter
from app.models.registry import model_registry
from app.models.sar_fusion.adapter import SARFusionAdapter


def register_default_models() -> None:
    """Register default specialist model adapters into global model_registry."""
    model_registry.register(GeoChatAdapter())
    model_registry.register(ChangeChatAdapter())
    model_registry.register(PrithviAdapter())
    model_registry.register(SARFusionAdapter())
    model_registry.register(BigEarthNetVLMAdapter())  # Differentiator #1
