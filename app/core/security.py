import os
import re
import uuid
from pathlib import Path
from app.core.config import settings


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames to prevent path traversal."""
    filename = Path(filename).name
    # Keep alphanumeric, underscores, hyphens, and single dots
    sanitized = re.sub(r"[^\w\.-]", "_", filename)
    return sanitized or f"upload_{uuid.uuid4().hex[:8]}"


def validate_file_extension(filename: str) -> bool:
    """Validate file extension against allowed extensions."""
    ext = Path(filename).suffix.lower()
    return ext in settings.ALLOWED_EXTENSIONS


def get_safe_file_path(base_dir: str, filename: str) -> Path:
    """Return safe absolute path within base_dir, preventing path traversal."""
    safe_name = sanitize_filename(filename)
    base_path = Path(base_dir).resolve()
    target_path = (base_path / safe_name).resolve()

    if not str(target_path).startswith(str(base_path)):
        raise ValueError("Path traversal attempt detected.")

    return target_path
