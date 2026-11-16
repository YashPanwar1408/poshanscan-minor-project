import os
import uuid
import httpx
from app.core.config import settings


def save_upload_file(
    image_bytes: bytes,
    original_filename: str = "image.jpg",
    content_type: str = "image/jpeg",
) -> str:
    """Persist an image to Supabase Storage when configured, otherwise use local disk."""
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4()}{ext}"

    if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY:
        object_path = f"scans/{unique_filename}"
        endpoint = (
            f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/"
            f"{settings.SUPABASE_STORAGE_BUCKET}/{object_path}"
        )
        headers = {
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": content_type or "application/octet-stream",
            "x-upsert": "false",
        }
        try:
            response = httpx.post(endpoint, content=image_bytes, headers=headers, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError("Could not save the screening image to Supabase Storage.") from exc
        return (
            f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/public/"
            f"{settings.SUPABASE_STORAGE_BUCKET}/{object_path}"
        )

    target_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    with open(target_path, "wb") as f:
        f.write(image_bytes)

    return target_path


def upload_url(image_path: str | None, base_url: str) -> str | None:
    """Return a browser URL for a stored image, preserving already-public URLs."""
    if not image_path:
        return None
    if image_path.startswith(("http://", "https://")):
        return image_path
    return f"{base_url.rstrip('/')}/uploads/{os.path.basename(image_path)}"
