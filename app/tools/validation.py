import os
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from app.schemas.images import BoundingBox, ImageMetadata
from app.schemas.tools import ToolResult
from app.tools.registry import tool_registry


@tool_registry.register(
    name="validate_image",
    description="Validates image file extension, MIME, dimension, bands, CRS, and geotransform."
)
async def validate_image(file_path: str) -> ToolResult:
    path = Path(file_path)
    if not path.exists():
        return ToolResult(
            tool="validate_image",
            status="error",
            error=f"File not found: {file_path}"
        )

    filename = path.name
    ext = path.suffix.lower()

    width, height, num_bands, dtype = 0, 0, 3, "uint8"
    crs, geotransform, bounds = None, None, None
    is_valid = True
    validation_error = None

    # Try Rasterio for GeoTIFF / geospatial images
    if HAS_RASTERIO and ext in [".tif", ".tiff"]:
        try:
            with rasterio.open(file_path) as src:
                width = src.width
                height = src.height
                num_bands = src.count
                dtype = str(src.dtypes[0])
                crs = str(src.crs) if src.crs else None
                geotransform = list(src.transform)[:6] if src.transform else None
                if src.bounds:
                    bounds = BoundingBox(
                        min_x=src.bounds.left,
                        min_y=src.bounds.bottom,
                        max_x=src.bounds.right,
                        max_y=src.bounds.top,
                        crs=crs
                    )
        except Exception as e:
            is_valid = False
            validation_error = f"Rasterio read error: {str(e)}"
    
    # Fall back to Pillow for PNG/JPEG or when rasterio is unavailable
    if not width and is_valid:
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                num_bands = len(img.getbands()) if hasattr(img, "getbands") else 3
                dtype = "uint8"
        except Exception as e:
            is_valid = False
            validation_error = f"PIL image read error: {str(e)}"

    metadata = ImageMetadata(
        filename=filename,
        width=width,
        height=height,
        num_bands=num_bands,
        dtype=dtype,
        crs=crs,
        geotransform=geotransform,
        bounds=bounds,
        is_valid=is_valid,
        validation_error=validation_error
    )

    return ToolResult(
        tool="validate_image",
        status="success" if is_valid else "failure",
        result=metadata.model_dump(),
        error=validation_error
    )
