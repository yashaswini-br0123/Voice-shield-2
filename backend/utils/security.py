import os
import re
import tempfile
from contextlib import contextmanager
from typing import Generator
from fastapi import HTTPException, UploadFile, status
from backend.config import settings


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes upload filename to prevent path traversal and unsafe character injection.
    """
    if not filename:
        return "uploaded_file.bin"
    basename = os.path.basename(filename)
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', basename)
    return sanitized


def get_media_type(filename: str) -> str:
    """
    Determines media type ('audio', 'image', 'video') based on file extension.
    """
    if not filename:
        return "audio"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in settings.IMAGE_EXTENSIONS:
        return "image"
    elif ext in settings.VIDEO_EXTENSIONS:
        return "video"
    return "audio"


def validate_uploaded_file(file: UploadFile, contents: bytes) -> str:
    """
    Validates file size limit (50MB max) and non-emptiness.
    Returns detected media type ('audio', 'image', 'video').
    """
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )
    
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB."
        )
        
    return get_media_type(file.filename or "")


@contextmanager
def save_temp_file(contents: bytes, filename: str) -> Generator[str, None, None]:
    """
    Context manager that saves uploaded bytes into a temporary file and guarantees
    automatic cleanup upon completion or failure.
    """
    sanitized = sanitize_filename(filename)
    base_name, ext = os.path.splitext(sanitized)
    ext = ext or ".bin"
    prefix = f"vs_{base_name}_"
    
    temp_dir = tempfile.gettempdir()
    fd, temp_path = tempfile.mkstemp(suffix=ext, prefix=prefix, dir=temp_dir)
    
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(contents)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
