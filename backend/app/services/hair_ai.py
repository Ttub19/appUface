"""
"สร้างภาพสมจริง" for the hair try-on: a generative image model redraws the
hair on the user's still photo so it looks like real hair instead of a 2D
cut-out.

Two providers, picked with HAIR_AI_PROVIDER (or automatically from whichever
credentials are set):

  cloudflare  (default, free) — Cloudflare Workers AI, Stable Diffusion 1.5
              inpainting. The browser sends a 512x512 crop around the head of
              the 2D try-on (= the shape and colour of the chosen style) plus a
              mask of the hair area; only the masked hair is repainted
              ("patch" mode), so the face stays exactly the user's own.
              Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN

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
CLOUDFLARE_MODEL = "@cf/runwayml/stable-diffusion-v1-5-inpainting"
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
                        env.get("GEMINI_IMAGE_MODEL", "").strip() or DEFAULT_GEMINI_MODEL)


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


def cloudflare_prompt(style_name: str, color_words: str) -> tuple[str, str]:
    prompt = (f"photo of a person with a {style_name} haircut, {color_words} hair, realistic natural hair strands, "
              "detailed hair texture, soft natural lighting, photorealistic, sharp focus")
    negative = ("cartoon, drawing, painting, illustration, wig, helmet, plastic, blurry, low quality, deformed, "
                "extra face, distorted face, text, watermark")
    return prompt, negative


def gemini_prompt(style_name: str, color_words: str) -> str:
    return (f"Edit the first photo: give the person the {style_name} hairstyle shown in the second image, "
            f"with {color_words} hair. Make the hair look completely real (natural strands, volume, lighting and "
            "shadows matching the photo), fitted to this person's head. Keep the face, identity, expression, skin, "
            "eyes, clothes, background, framing and image size exactly the same; change only the hair. "
            "Return only the edited photo.")


def cloudflare_payload(guide_png: bytes, mask_png: bytes, style_name: str, color_words: str, seed: int = 42) -> dict:
    prompt, negative = cloudflare_prompt(style_name, color_words)
    return {
        "prompt": prompt, "negative_prompt": negative,
        "image": list(guide_png), "mask": list(mask_png),   # Workers AI takes encoded image files as byte arrays
        "width": PATCH_SIZE, "height": PATCH_SIZE,
        # the guide already has the right hair shape and colour: keep it, repaint texture + lighting
        "strength": .6, "num_steps": 20, "guidance": 7.5, "seed": seed,
    }


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
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
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
        url = f"https://api.cloudflare.com/client/v4/accounts/{cfg.cloudflare_account_id}/ai/run/{CLOUDFLARE_MODEL}"
        status_code, ctype, body = _post(url, {"Authorization": f"Bearer {cfg.cloudflare_api_token}"},
                                         cloudflare_payload(guide, mask, style_name, color_words))
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
