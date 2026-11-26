"""
"สร้างภาพสมจริง" for the hair try-on: a generative image model redraws the
hair on the user's still photo so it looks like real hair instead of a 2D
cut-out.

Two providers, picked with HAIR_AI_PROVIDER (or automatically from whichever
credentials are set):

  cloudflare  (default, free) — Cloudflare Workers AI, FLUX.2 [klein] 9B image
              editing (Stable Diffusion 1.5 inpainting, used before, was retired
              by Cloudflare on 2026-11-20). The browser sends a 512x512 crop
              around the head of the 2D try-on (= the shape and colour of the
              chosen style) plus a mask of the hair area. FLUX.2 has no mask
              input, so it edits the whole crop, and the browser pastes back only
              the masked hair ("patch" mode) — the face stays the user's own.
              Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN,
                   optional CLOUDFLARE_IMAGE_MODEL

  gemini      (better quality, paid / Google Cloud trial credit) — Gemini image
              editing. Gets the original photo + the chosen hairstyle picture
              as a reference and returns the whole edited photo ("full" mode).
              Env: GEMINI_API_KEY, optional GEMINI_IMAGE_MODEL

Each signed-in member gets HAIR_AI_DAILY_LIMIT generations per day
(Asia/Bangkok day, default 5); only successful generations count.

Everything below except call_provider() is pure, so it is unit-tested without
network access.
"""
import base64
import binascii
import io
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

BANGKOK = timezone(timedelta(hours=7))
DEFAULT_DAILY_LIMIT = 5
# FLUX.2 models take multipart/form-data (prompt, width, height, input_image_0..); SD 1.5 inpainting was retired 2026-11-20
CLOUDFLARE_MODEL = "@cf/black-forest-labs/flux-2-klein-9b"
DEFAULT_GEMINI_MODEL = "gemini-nano-banana-2.1"
PATCH_SIZE = 512                       # SD 1.5 works at 512 px
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
TIMEOUT_S = 90

PROVIDER_LABELS = {"cloudflare": "Cloudflare Workers AI", "gemini": "Google Gemini"}

# English names for the colour presets of the try-on (frontend HAIR_COLORS ids)
HAIR_COLOR_WORDS = {
    "black": "jet black", "dark-brown": "dark brown", "brown": "medium brown", "ash": "ash grey brown",
    "blonde": "golden blonde", "wine": "burgundy wine red",
}


class HairAIError(RuntimeError):
    """Provider failed or returned something unusable; message is shown to the user."""


@dataclass(frozen=True)
class HairAIConfig:
    provider: Optional[str]            # None = feature off (no credentials)
    daily_limit: int
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    cloudflare_model: str = CLOUDFLARE_MODEL

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    @property
    def mode(self) -> Optional[str]:
        return {"cloudflare": "patch", "gemini": "full"}.get(self.provider)


def load_config(env=os.environ) -> HairAIConfig:
    cf_id, cf_token = env.get("CLOUDFLARE_ACCOUNT_ID", "").strip(), env.get("CLOUDFLARE_API_TOKEN", "").strip()
    gem_key = env.get("GEMINI_API_KEY", "").strip()
    has = {"cloudflare": bool(cf_id and cf_token), "gemini": bool(gem_key)}
    wanted = env.get("HAIR_AI_PROVIDER", "").strip().lower()
    if wanted in has:
        provider = wanted if has[wanted] else None
        if not has[wanted]:
            logger.warning("HAIR_AI_PROVIDER=%s but its credentials are missing — realistic hair is off", wanted)
    elif wanted in ("", "auto"):
        provider = "cloudflare" if has["cloudflare"] else ("gemini" if has["gemini"] else None)
    else:   # "off" or anything else
        provider = None
    try:
        limit = max(0, int(env.get("HAIR_AI_DAILY_LIMIT", DEFAULT_DAILY_LIMIT)))
    except ValueError:
        limit = DEFAULT_DAILY_LIMIT
    return HairAIConfig(provider, limit, cf_id, cf_token, gem_key,
                        env.get("GEMINI_IMAGE_MODEL", "").strip() or DEFAULT_GEMINI_MODEL,
                        env.get("CLOUDFLARE_IMAGE_MODEL", "").strip() or CLOUDFLARE_MODEL)


def day_start_utc(now: Optional[datetime] = None) -> datetime:
    """Start of today in Bangkok, as an aware UTC datetime (the quota resets at local midnight)."""
    now = (now or datetime.now(timezone.utc)).astimezone(BANGKOK)
    return now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)


def decode_image_data_url(data_url: str, field: str) -> tuple[bytes, str]:
    """data:image/png;base64,... -> (raw bytes, mime). Checks it really is an image."""
    from PIL import Image, UnidentifiedImageError

    if not data_url or not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError(f"{field}: ต้องเป็นรูปแบบ data URL ของรูปภาพ")
    head, payload = data_url.split(",", 1)
    mime = head[5:].split(";")[0]
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field}: รูปเสียหาย") from exc
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{field}: รูปใหญ่เกินไป")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{field}: ไม่ใช่ไฟล์รูปภาพ") from exc
    return raw, mime


def image_size(raw: bytes) -> tuple[int, int]:
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.size


def describe_color(color_id: Optional[str], color_rgb: Optional[list]) -> str:
    if color_id in HAIR_COLOR_WORDS:
        return HAIR_COLOR_WORDS[color_id]
    if color_rgb and len(color_rgb) == 3:      # "เหมือนผมฉัน" / style colour: describe the sampled RGB
        r, g, b = (int(v) for v in color_rgb)
        lum = .299 * r + .587 * g + .114 * b
        tone = "black" if lum < 45 else "dark brown" if lum < 85 else "brown" if lum < 130 else "light brown" if lum < 170 else "blonde"
        return f"natural {tone}"
    return "natural"


def cloudflare_prompt(style_name: str, color_words: str) -> str:
    """Edit instruction for FLUX.2: the input crop already has the right hair shape and colour (2D try-on)."""
    return (f"Edit this photo: turn the {style_name} hairstyle on the person's head into completely real, natural "
            f"{color_words} hair with individual strands, volume, soft natural shadows and lighting that match the photo. "
            "Keep the hair's shape, length, parting and outline exactly as in the image. "
            "Do not change the face, eyes, skin, expression, ears, clothes, background or framing. "
            "Photorealistic photo, sharp focus, no wig, no helmet, no drawing, no text.")


def gemini_prompt(style_name: str, color_words: str) -> str:
    return (f"Edit the first photo: give the person the {style_name} hairstyle shown in the second image, "
            f"with {color_words} hair. Make the hair look completely real (natural strands, volume, lighting and "
            "shadows matching the photo), fitted to this person's head. Keep the face, identity, expression, skin, "
            "eyes, clothes, background, framing and image size exactly the same; change only the hair. "
            "Return only the edited photo.")


def cloudflare_form(guide_png: bytes, style_name: str, color_words: str) -> list:
    """multipart/form-data fields for FLUX.2 on Workers AI: (name, value, filename, content type).
    The mask is not sent (FLUX.2 has no mask input); the browser keeps only the masked hair of the result."""
    return [
        ("prompt", cloudflare_prompt(style_name, color_words), None, None),
        ("width", str(PATCH_SIZE), None, None),
        ("height", str(PATCH_SIZE), None, None),
        ("input_image_0", guide_png, "guide.png", "image/png"),
    ]


def encode_multipart(fields: list, boundary: Optional[str] = None) -> tuple[bytes, str]:
    """Build a multipart/form-data body (stdlib only). Returns (body, content type header)."""
    import uuid
    boundary = boundary or f"ushape-{uuid.uuid4().hex}"
    out = io.BytesIO()
    for name, value, filename, ctype in fields:
        out.write(f"--{boundary}\r\n".encode())
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
        out.write((disp + "\r\n").encode())
        if ctype:
            out.write(f"Content-Type: {ctype}\r\n".encode())
        out.write(b"\r\n")
        out.write(value if isinstance(value, bytes) else str(value).encode("utf-8"))
        out.write(b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def gemini_payload(model: str, photo: bytes, photo_mime: str, reference: bytes, ref_mime: str, prompt: str) -> dict:
    return {
        "model": model,
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "mime_type": photo_mime, "data": base64.b64encode(photo).decode()},
            {"type": "image", "mime_type": ref_mime, "data": base64.b64encode(reference).decode()},
        ],
    }


def _sniff_mime(raw: bytes) -> Optional[str]:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def parse_cloudflare_response(content_type: str, body: bytes) -> tuple[bytes, str]:
    """Workers AI answers image models with the raw PNG; errors (and some models) come as JSON."""
    mime = _sniff_mime(body)
    if mime:
        return body, mime
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HairAIError("บริการ AI ตอบกลับในรูปแบบที่อ่านไม่ได้") from exc
    if isinstance(data, dict) and data.get("success") is False:
        msg = "; ".join(str(e.get("message", e)) for e in data.get("errors") or []) or "unknown error"
        raise HairAIError(f"Cloudflare ตอบกลับข้อผิดพลาด: {msg}")
    found = _find_image_b64(data)
    if not found:
        raise HairAIError("บริการ AI ไม่ได้ส่งรูปกลับมา")
    return found


def _find_image_b64(node) -> Optional[tuple[bytes, str]]:
    """Last base64 image anywhere in a JSON response (Gemini Interactions / generateContent / Workers AI shapes)."""
    found = None
    if isinstance(node, dict):
        for key in ("inlineData", "inline_data"):
            blob = node.get(key)
            if isinstance(blob, dict) and blob.get("data"):
                found = _decode_b64_image(blob["data"], blob.get("mimeType") or blob.get("mime_type")) or found
        if node.get("type") == "image" and isinstance(node.get("data"), str):
            found = _decode_b64_image(node["data"], node.get("mime_type") or node.get("mimeType")) or found
        if isinstance(node.get("image"), str):
            found = _decode_b64_image(node["image"], None) or found
        for value in node.values():
            if isinstance(value, (dict, list)):
                found = _find_image_b64(value) or found
    elif isinstance(node, list):
        for item in node:
            found = _find_image_b64(item) or found
    return found


def _decode_b64_image(data: str, mime: Optional[str]) -> Optional[tuple[bytes, str]]:
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):
        return None
    sniffed = _sniff_mime(raw)
    return (raw, sniffed or mime or "image/png") if sniffed else None


def parse_gemini_response(body: bytes) -> tuple[bytes, str]:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HairAIError("Gemini ตอบกลับในรูปแบบที่อ่านไม่ได้") from exc
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise HairAIError(f"Gemini ตอบกลับข้อผิดพลาด: {err.get('message', err) if isinstance(err, dict) else err}")
    found = _find_image_b64(data)
    if not found:
        # e.g. the safety filter refused, so the model answered with text only
        raise HairAIError("Gemini ไม่ได้สร้างรูปกลับมา (อาจถูกระบบความปลอดภัยปฏิเสธ) ลองรูปอื่นหรือทรงอื่น")
    return found


def _post(url: str, headers: dict, payload: dict) -> tuple[int, str, bytes]:
    return _post_raw(url, {"Content-Type": "application/json", **headers}, json.dumps(payload).encode("utf-8"))


def _post_raw(url: str, headers: dict, data: bytes) -> tuple[int, str, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as res:
            return res.status, res.headers.get("Content-Type", ""), res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", "") if exc.headers else "", exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HairAIError("เชื่อมต่อบริการ AI ไม่ได้ กรุณาลองใหม่อีกครั้ง") from exc


def call_provider(cfg: HairAIConfig, *, style_name: str, color_words: str,
                  guide: Optional[bytes] = None, mask: Optional[bytes] = None,
                  photo: Optional[tuple[bytes, str]] = None, reference: Optional[tuple[bytes, str]] = None) -> tuple[bytes, str]:
    """Blocking network call (run it in a thread). Returns (image bytes, mime)."""
    if cfg.provider == "cloudflare":
        url = f"https://api.cloudflare.com/client/v4/accounts/{cfg.cloudflare_account_id}/ai/run/{cfg.cloudflare_model}"
        body_bytes, content_type = encode_multipart(cloudflare_form(guide, style_name, color_words))
        status_code, ctype, body = _post_raw(url, {"Authorization": f"Bearer {cfg.cloudflare_api_token}",
                                                   "Content-Type": content_type}, body_bytes)
        if status_code >= 400 and not _sniff_mime(body):
            logger.warning("Cloudflare AI HTTP %s: %s", status_code, body[:300])
            parse_cloudflare_response(ctype, body)          # raises with the provider's message when it's JSON
            raise HairAIError(f"Cloudflare ตอบกลับข้อผิดพลาด (HTTP {status_code})")
        return parse_cloudflare_response(ctype, body)
    if cfg.provider == "gemini":
        prompt = gemini_prompt(style_name, color_words)
        url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        status_code, _, body = _post(url, {"x-goog-api-key": cfg.gemini_api_key},
                                     gemini_payload(cfg.gemini_model, photo[0], photo[1], reference[0], reference[1], prompt))
        if status_code >= 400:
            logger.warning("Gemini HTTP %s: %s", status_code, body[:300])
            parse_gemini_response(body)
            raise HairAIError(f"Gemini ตอบกลับข้อผิดพลาด (HTTP {status_code})")
        return parse_gemini_response(body)
    raise HairAIError("ยังไม่ได้ตั้งค่าบริการ AI บนเซิร์ฟเวอร์")


def to_data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
