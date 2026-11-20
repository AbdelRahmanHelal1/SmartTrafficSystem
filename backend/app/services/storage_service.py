from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from app.config import settings

ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
MAX_SIZE = 5 * 1024 * 1024  # 5MB


async def save_image(file: UploadFile, folder: str) -> str:
    """يحفظ الصورة في storage/<folder> ويرجّع المسار النسبي."""
    ext = ALLOWED_TYPES.get(file.content_type)
    if ext is None:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are allowed")

    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 5MB)")

    target_dir: Path = settings.storage_dir / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4().hex}{ext}"
    (target_dir / filename).write_bytes(content)

    return f"{folder}/{filename}"