import os
import uuid
import httpx
from urllib.parse import quote, urlparse
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
    """Return a browser URL for a stored image.

    Supabase objects are served through the backend proxy so the UI remains
    reliable when the bucket is private or its public policy changes.
    """
    if not image_path:
        return None
    if image_path.startswith(("http://", "https://")):
        parsed = urlparse(image_path)
        marker = "/storage/v1/object/"
        if marker in parsed.path:
            object_path = parsed.path.split(marker, 1)[1]
            parts = object_path.split("/", 1)
            if parts and parts[0] in {"public", "sign"} and len(parts) == 2:
                object_path = parts[1]
                parts = object_path.split("/", 1)
            if len(parts) == 2:
                bucket, path = parts
                if bucket == settings.SUPABASE_STORAGE_BUCKET:
                    return f"{base_url.rstrip('/')}/media/{quote(path, safe='/')}"
        return image_path
    return f"{base_url.rstrip('/')}/uploads/{os.path.basename(image_path)}"


def fetch_storage_object(object_path: str) -> tuple[bytes, str]:
    """Fetch one Supabase Storage object for the backend media proxy."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Supabase Storage is not configured.")
    endpoint = (
        f"{settings.SUPABASE_URL.rstrip('/')}/storage/v1/object/"
        f"{settings.SUPABASE_STORAGE_BUCKET}/{object_path.lstrip('/')}"
    )
    headers = {
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
    }
    try:
        response = httpx.get(endpoint, headers=headers, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not read the screening image from Supabase Storage.") from exc
    return response.content, response.headers.get("content-type", "application/octet-stream")
