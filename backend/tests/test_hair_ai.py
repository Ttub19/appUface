"""
"สร้างภาพสมจริง" (AI hair): provider selection, daily window, request payloads
and response parsing for Cloudflare Workers AI and Gemini — no network.
Run: python -m unittest discover -s tests
"""
import base64
import io
import json
import unittest
from datetime import datetime, timezone

from PIL import Image

from app.services import hair_ai


def png_bytes(size=(512, 512), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def data_url(raw, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


class Config(unittest.TestCase):
    CF = {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok"}
    GEM = {"GEMINI_API_KEY": "key"}

    def test_off_without_credentials(self):
        cfg = hair_ai.load_config({})
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.mode)
        self.assertEqual(cfg.daily_limit, 5)

    def test_auto_prefers_free_cloudflare(self):
        self.assertEqual(hair_ai.load_config({**self.CF, **self.GEM}).provider, "cloudflare")
        self.assertEqual(hair_ai.load_config(self.GEM).provider, "gemini")
        self.assertEqual(hair_ai.load_config(self.CF).mode, "full")
        self.assertEqual(hair_ai.load_config({**self.CF, "CLOUDFLARE_HAIR_MODE": "patch"}).mode, "patch")
        self.assertEqual(hair_ai.load_config(self.GEM).mode, "full")

    def test_explicit_provider_and_limit(self):
        cfg = hair_ai.load_config({**self.CF, **self.GEM, "HAIR_AI_PROVIDER": "gemini", "HAIR_AI_DAILY_LIMIT": "3",
                                   "GEMINI_IMAGE_MODEL": "gemini-3-pro-image"})
        self.assertEqual((cfg.provider, cfg.daily_limit, cfg.gemini_model), ("gemini", 3, "gemini-3-pro-image"))
        self.assertFalse(hair_ai.load_config({**self.CF, "HAIR_AI_PROVIDER": "gemini"}).enabled)   # no key -> off
        self.assertFalse(hair_ai.load_config({**self.CF, "HAIR_AI_PROVIDER": "off"}).enabled)
        self.assertEqual(hair_ai.load_config({**self.CF, "HAIR_AI_DAILY_LIMIT": "x"}).daily_limit, 5)


class DayWindow(unittest.TestCase):
    def test_bangkok_midnight(self):
        # 2026-10-29 01:30 in Bangkok = 2026-10-28 18:30 UTC -> day starts 2026-10-28 17:00 UTC
        start = hair_ai.day_start_utc(datetime(2026, 10, 28, 18, 30, tzinfo=timezone.utc))
        self.assertEqual(start, datetime(2026, 10, 28, 17, 0, tzinfo=timezone.utc))
        start = hair_ai.day_start_utc(datetime(2026, 10, 29, 16, 59, tzinfo=timezone.utc))
        self.assertEqual(start, datetime(2026, 10, 28, 17, 0, tzinfo=timezone.utc))


class Inputs(unittest.TestCase):
    def test_decode_checks_real_image(self):
        raw, mime = hair_ai.decode_image_data_url(data_url(png_bytes()), "guide")
        self.assertEqual(mime, "image/png")
        self.assertEqual(hair_ai.image_size(raw), (512, 512))
        for bad in ("nope", "data:text/plain;base64,aGk=", data_url(b"not an image")):
            with self.assertRaises(ValueError):
                hair_ai.decode_image_data_url(bad, "guide")

    def test_colour_words(self):
        self.assertEqual(hair_ai.describe_color("blonde", None), "golden blonde")
        self.assertEqual(hair_ai.describe_color("mine", [20, 18, 16]), "natural black")
        self.assertEqual(hair_ai.describe_color("mine", [180, 150, 110]), "natural light brown")
        self.assertEqual(hair_ai.describe_color(None, None), "natural")

    def test_cloudflare_form_is_multipart_for_flux2(self):
        g = png_bytes()
        fields = hair_ai.cloudflare_form(g, "Buzz Cut", "jet black")
        by_name = {f[0]: f for f in fields}
        self.assertEqual(by_name["input_image_0"][1], g)
        self.assertEqual(by_name["input_image_0"][3], "image/png")
        self.assertIn("Buzz Cut", by_name["prompt"][1])
        self.assertIn("jet black", by_name["prompt"][1])
        self.assertEqual((by_name["width"][1], by_name["height"][1]), ("512", "512"))
        body, ctype = hair_ai.encode_multipart(fields, boundary="B")
        self.assertEqual(ctype, "multipart/form-data; boundary=B")
        self.assertIn(b'name="input_image_0"; filename="guide.png"', body)
        self.assertIn(g, body)
        self.assertTrue(body.endswith(b"--B--\r\n"))
        # the parts parse back with the stdlib email parser
        from email.parser import BytesParser
        msg = BytesParser().parsebytes(f"Content-Type: {ctype}\r\n\r\n".encode() + body)
        parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True) for p in msg.get_payload()}
        self.assertEqual(parts["input_image_0"], g)
        self.assertEqual(parts["width"], b"512")

    def test_cloudflare_full_form_sends_photo_and_hairstyle(self):
        photo = png_bytes((900, 1200))
        ref = png_bytes((300, 300), (200, 150, 100))
        fields = hair_ai.cloudflare_full_form((photo, "image/png"), (ref, "image/png"), "Wolf Cut", "dark brown")
        by_name = {f[0]: f for f in fields}
        self.assertEqual(by_name["input_image_0"][1], photo)
        self.assertEqual(by_name["input_image_1"][1], ref)
        self.assertIn("Wolf Cut", by_name["prompt"][1])
        self.assertIn("exactly the same", by_name["prompt"][1])
        w, h = int(by_name["width"][1]), int(by_name["height"][1])
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertLessEqual(max(w, h), 1024)
        self.assertAlmostEqual(w / h, 900 / 1200, delta=0.03)

    def test_flux_size_limits(self):
        self.assertEqual(hair_ai.flux_size(1024, 768), (1024, 768))
        self.assertEqual(hair_ai.flux_size(4000, 3000), (1024, 768))
        self.assertEqual(hair_ai.flux_size(100, 2000), (256, 1024))   # clamped to the model's minimum

    def test_cloudflare_model_default_and_override(self):
        cf = {"CLOUDFLARE_ACCOUNT_ID": "acc", "CLOUDFLARE_API_TOKEN": "tok"}
        self.assertEqual(hair_ai.load_config(cf).cloudflare_model, "@cf/black-forest-labs/flux-2-klein-9b")
        self.assertEqual(hair_ai.load_config({**cf, "CLOUDFLARE_IMAGE_MODEL": "@cf/black-forest-labs/flux-2-klein-4b"}).cloudflare_model,
                         "@cf/black-forest-labs/flux-2-klein-4b")

    def test_gemini_payload(self):
        p = hair_ai.gemini_payload("m", b"PHOTO", "image/jpeg", b"REF", "image/png", "do it")
        self.assertEqual(p["model"], "m")
        self.assertEqual([i["type"] for i in p["input"]], ["text", "image", "image"])
        self.assertEqual(base64.b64decode(p["input"][1]["data"]), b"PHOTO")
        self.assertEqual(p["input"][2]["mime_type"], "image/png")
        self.assertIn("exactly the same", hair_ai.gemini_prompt("Pixie Cut", "golden blonde"))


class Responses(unittest.TestCase):
    def test_cloudflare_raw_png(self):
        img = png_bytes()
        self.assertEqual(hair_ai.parse_cloudflare_response("image/png", img), (img, "image/png"))

    def test_cloudflare_json_image_and_error(self):
        img = png_bytes()
        body = json.dumps({"result": {"image": base64.b64encode(img).decode()}, "success": True}).encode()
        self.assertEqual(hair_ai.parse_cloudflare_response("application/json", body)[0], img)
        err = json.dumps({"success": False, "errors": [{"message": "Authentication error"}]}).encode()
        with self.assertRaisesRegex(hair_ai.HairAIError, "Authentication error"):
            hair_ai.parse_cloudflare_response("application/json", err)

    def test_gemini_interactions_and_generate_content_shapes(self):
        img = png_bytes()
        b64 = base64.b64encode(img).decode()
        interactions = {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"},
                                                                      {"type": "image", "mime_type": "image/png", "data": b64}]}]}
        legacy = {"candidates": [{"content": {"parts": [{"text": "ok"}, {"inlineData": {"mimeType": "image/png", "data": b64}}]}}]}
        for body in (interactions, legacy):
            self.assertEqual(hair_ai.parse_gemini_response(json.dumps(body).encode()), (img, "image/png"))

    def test_gemini_text_only_or_error(self):
        with self.assertRaises(hair_ai.HairAIError):
            hair_ai.parse_gemini_response(json.dumps({"steps": [{"content": [{"type": "text", "text": "sorry"}]}]}).encode())
        with self.assertRaisesRegex(hair_ai.HairAIError, "API key not valid"):
            hair_ai.parse_gemini_response(json.dumps({"error": {"code": 400, "message": "API key not valid"}}).encode())


if __name__ == "__main__":
    unittest.main()
