from typing import Any, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min_x: float = Field(..., description="Min X coordinate / longitude / pixel min_x")
    min_y: float = Field(..., description="Min Y coordinate / latitude / pixel min_y")
    max_x: float = Field(..., description="Max X coordinate / longitude / pixel max_x")
    max_y: float = Field(..., description="Max Y coordinate / latitude / pixel max_y")
    crs: Optional[str] = Field(None, description="Coordinate Reference System (e.g. EPSG:4326)")


class ImageMetadata(BaseModel):
    filename: str
    width: int
    height: int
    num_bands: int
    dtype: str
    crs: Optional[str] = None
    geotransform: Optional[list[float]] = None
    bounds: Optional[BoundingBox] = None
    acquisition_date: Optional[str] = None
    sensor: Optional[str] = None
    spatial_resolution_m: Optional[float] = None
    modality: str = Field("unknown", description="optical, multispectral, sar, unknown")
    modality_confidence: float = Field(0.0, ge=0.0, le=1.0)
    modality_source: str = Field("default", description="metadata, band_heuristic, model_inference")
    is_valid: bool = True
    validation_error: Optional[str] = None


class ImageInputConfig(BaseModel):
    configuration_type: str = Field(
        ...,
        description="single_image, bitemporal_pair, optical_sar_pair, unknown"
    )
    image_count: int
    images: list[ImageMetadata]
    temporal_compatible: bool = True
    spatial_compatible: bool = True
    compatibility_reason: Optional[str] = None
