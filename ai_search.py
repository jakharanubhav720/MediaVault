import json
import mimetypes
import os
import time
from pathlib import Path
 
import numpy as np
from PIL import Image, ImageOps
 
Image.MAX_IMAGE_PIXELS = 25_000_000  # cap decoded image memory on small instances
 
# This branch is intended for local development on a machine with sufficient RAM.
# Keep the Render Free deployment on main, where local vision models remain disabled.
_EMBED_BACKEND_REQUESTED = os.getenv("IMAGE_EMBED_BACKEND", "siglip2").strip().lower()
# Treat the previous "clip" setting as SigLIP 2 during migration, so existing
# local .env files continue using a semantic model instead of silently falling back.
_SIGLIP2_REQUESTED = _EMBED_BACKEND_REQUESTED in ("siglip2", "clip")
_SIGLIP2_ALLOWED = os.getenv("ENABLE_SIGLIP2", "true").strip().lower() in ("1", "true", "yes", "on")
IMAGE_BACKEND = "siglip2" if _SIGLIP2_REQUESTED and _SIGLIP2_ALLOWED else "visual"
SIGLIP2_MODEL_ID = os.getenv("SIGLIP2_MODEL_ID", "google/siglip2-base-patch16-256")
 
# ---------------------------------------------------------------- image vectors
 
 
def _normalize(block, center=True):
    block = np.asarray(block, dtype=np.float64).ravel()
    if center:
        block = block - block.mean()
    norm = np.linalg.norm(block)
    return block / norm if norm > 1e-12 else np.zeros_like(block)
 
 
def _open_rgb(path):
    try:
        with Image.open(path) as source:
            # Ask JPEG decoders to avoid loading the full-resolution image.
            source.draft("RGB", (512, 512))
            # Resize before color conversion so large uploads do not create full-size RGB copies.
            source.thumbnail((512, 512), Image.Resampling.LANCZOS)
            img = ImageOps.exif_transpose(source)
            return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Not a readable image ({type(exc).__name__})") from exc
 
 
def _visual_vector(path):
    img = _open_rgb(path)
 
    # 1) colour layout: 8x8 grid of average colours, brightness-insensitive
    layout = np.asarray(img.resize((8, 8), Image.BOX), dtype=np.float64) / 255.0
    layout = layout - layout.mean(axis=(0, 1), keepdims=True)
 
    small = img.resize((64, 64), Image.BILINEAR)
 
    # 2) joint hue / saturation / value histogram (Hellinger-normalised)
    hsv = np.asarray(small.convert("HSV"), dtype=np.int64)
    h_bin = np.minimum(hsv[..., 0] * 12 // 256, 11)
    s_bin = np.minimum(hsv[..., 1] * 3 // 256, 2)
    v_bin = np.minimum(hsv[..., 2] * 3 // 256, 2)
    index = (h_bin * 3 + s_bin) * 3 + v_bin
    hist = np.bincount(index.ravel(), minlength=12 * 3 * 3).astype(np.float64)
    hist = np.sqrt(hist / hist.sum())
 
    # 3) edge structure: gradient-orientation histogram over a 4x4 grid
    gray = np.asarray(small.convert("L"), dtype=np.float64)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    orient_bin = np.minimum((orientation / np.pi * 8).astype(np.int64), 7)
    cells = []
    for row in range(4):
        for col in range(4):
            sl = (slice(row * 16, row * 16 + 16), slice(col * 16, col * 16 + 16))
            cells.append(np.bincount(orient_bin[sl].ravel(), weights=magnitude[sl].ravel(), minlength=8))
    hog = np.sqrt(np.concatenate(cells) / (np.sum(cells) + 1e-9))
 
    vector = np.concatenate([
        0.35 * _normalize(layout),
        0.30 * _normalize(hist),
        0.35 * _normalize(hog),
    ])
    vector = _normalize(vector, center=False)
    return {"model": "visual-v1", "vec": [round(float(x), 5) for x in vector]}
 
 
_siglip2_model = None
_siglip2_processor = None
_siglip2_device = None


def _load_siglip2():
    """Load SigLIP 2 once, preferring Apple Silicon MPS when available."""
    global _siglip2_model, _siglip2_processor, _siglip2_device
    if _siglip2_model is not None:
        return _siglip2_model, _siglip2_processor, _siglip2_device

    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "SigLIP 2 requires torch and transformers. Install requirements.txt."
        ) from exc

    _siglip2_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(
        f"Loading SigLIP 2 model {SIGLIP2_MODEL_ID} on {_siglip2_device}...",
        flush=True,
    )
    _siglip2_processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
    _siglip2_model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID).to(_siglip2_device)
    _siglip2_model.eval()
    print("SigLIP 2 model loaded successfully.", flush=True)
    return _siglip2_model, _siglip2_processor, _siglip2_device


def _siglip2_image_vector(path):
    """Encode one image into SigLIP 2's shared image/text embedding space."""
    import torch

    model, processor, device = _load_siglip2()
    image = _open_rgb(path)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        # SigLIP 2 exposes dedicated feature methods; pass pixel_values explicitly.
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
        vector = features[0].detach().float().cpu().numpy()

    vector = _normalize(vector, center=False)
    return {
        "model": "siglip2-base-patch16-256",
        "vec": [round(float(x), 6) for x in vector],
    }


def siglip2_text_vector(text):
    """Encode text in the same vector space as SigLIP 2 image embeddings."""
    import torch

    text = (text or "").strip()
    if not text:
        raise ValueError("Enter a search description.")

    model, processor, device = _load_siglip2()
    # AutoProcessor applies SigLIP 2's expected text normalization and padding.
    inputs = processor(
        text=[text[:500]],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        # SigLIP 2 requires token IDs (and attention mask) for text features.
        features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        vector = features[0].detach().float().cpu().numpy()

    vector = _normalize(vector, center=False)
    return {
        "model": "siglip2-base-patch16-256",
        "vec": [round(float(x), 6) for x in vector],
    }


def image_vector(path):
    """Embedding record {'model': ..., 'vec': [...]} for one image file."""
    if IMAGE_BACKEND == "siglip2":
        return _siglip2_image_vector(path)
    return _visual_vector(path)


def current_image_model():
    return "siglip2-base-patch16-256" if IMAGE_BACKEND == "siglip2" else "visual-v1"


def current_text_model():
    if IMAGE_BACKEND == "siglip2":
        return "siglip2-base-patch16-256"
    return TEXT_MODEL if gemini_enabled() else "keywords"


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else 0.0
 
 
# ------------------------------------------------- optional Gemini text features
 
TEXT_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
VISION_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash").split(",")
    if m.strip()
]
TEXT_DIMS = 768
TEXT_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.55"))
TIMEOUT_MS = int(os.getenv("AI_TIMEOUT_MS", "8000"))
# Upload never waits longer than this for the optional Gemini description.
DESCRIBE_BUDGET_S = float(os.getenv("AI_DESCRIBE_BUDGET_S", "4"))
_client = None
 
 
def gemini_enabled():
    return bool(os.getenv("GEMINI_API_KEY"))
 
 
def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client
 
 
def text_embedding(text):
    text = (text or "").strip()
    if not text or not gemini_enabled():
        return None
    from google.genai import types
    result = _get_client().models.embed_content(
        model=TEXT_MODEL, contents=text[:4000],
        config=types.EmbedContentConfig(output_dimensionality=TEXT_DIMS),
    )
    return [round(float(x), 5) for x in result.embeddings[0].values]
 
 
def describe_image(path):
    from google.genai import types
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
    prompt = "Describe this image in 2-3 plain sentences: main objects, colors, setting and actions."
    errors = []
    started = time.monotonic()
    for model in VISION_MODELS:
        if time.monotonic() - started > DESCRIBE_BUDGET_S:
            errors.append("time budget exhausted")
            break
        try:
            response = _get_client().models.generate_content(model=model, contents=[part, prompt])
            text = (response.text or "").strip()
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {exc!r}"[:300])
    raise RuntimeError("All vision models failed: " + " | ".join(errors))
 
 
# ------------------------------------------------------------- stored records
 
 
def build_record(path, media_type, filename):
    """Return (description, record). Image matching never depends on Gemini."""
    record = {"v": 2}
    description = ""
    if media_type == "Image":
        record["img"] = image_vector(path)  # raises ValueError for unreadable images
        if gemini_enabled():
            try:
                started = time.monotonic()
                description = describe_image(path)
                vec = text_embedding(description) if time.monotonic() - started < DESCRIBE_BUDGET_S else None
                if vec:
                    record["txt"] = {"model": TEXT_MODEL, "vec": vec}
            except Exception as exc:
                print(f"Optional AI description skipped: {exc!r}", flush=True)
    elif gemini_enabled():
        try:
            vec = text_embedding(filename)
            if vec:
                record["txt"] = {"model": TEXT_MODEL, "vec": vec}
        except Exception as exc:
            print(f"Optional text embedding skipped: {exc!r}", flush=True)
    return description, record
 
 
def load_record(raw):
    """Parse a stored embedding column into {'img': ..., 'txt': ...} (never raises)."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, list):  # legacy format from earlier versions: a bare text vector
        return {"txt": {"model": "legacy", "vec": data}}
    return {}
 
 
# ----------------------------------------------------------- natural-language search
 
 
def search_media(query, rows):
    """Rank photos against SigLIP 2 text embeddings; retain keyword fallback for other media."""
    query = (query or "").strip()
    if not query:
        return []

    words = [w for w in query.casefold().split() if len(w) > 2]
    scored = []

    if IMAGE_BACKEND == "siglip2":
        query_record = siglip2_text_vector(query)
        for row in rows:
            text = f"{row['filename']} {row['description']}".casefold()
            keyword = bool(words) and any(w in text for w in words)
            entry = load_record(row["embedding"]).get("img")
            semantic = 0.0
            if row["media_type"] == "Image" and entry and entry.get("model") == query_record["model"] and entry.get("vec"):
                semantic = cosine_similarity(query_record["vec"], entry["vec"])
            # Cosine similarity is a ranking signal, not a calibrated probability.
            min_score = float(os.getenv("SIGLIP2_TEXT_MIN_SCORE", "0.18"))
            if (row["media_type"] == "Image" and semantic >= min_score) or keyword:
                item = dict(row)
                item["similarity"] = round(semantic, 4)
                item["match_percent"] = round(max(0.0, semantic) * 100, 1)
                scored.append((semantic + (0.12 if keyword else 0.0), item))
    else:
        query_vector = None
        if gemini_enabled():
            try:
                query_vector = text_embedding(query)
            except Exception as exc:
                print(f"Query embedding failed, using keyword search: {exc!r}", flush=True)

        for row in rows:
            text = f"{row['filename']} {row['description']}".casefold()
            keyword = bool(words) and any(w in text for w in words)
            semantic = 0.0
            entry = load_record(row["embedding"]).get("txt")
            if query_vector and entry and entry.get("model") in (TEXT_MODEL, "legacy"):
                semantic = cosine_similarity(query_vector, entry.get("vec"))
            if semantic >= TEXT_MIN_SCORE or keyword:
                item = dict(row)
                item["similarity"] = round(semantic, 4)
                scored.append((semantic + (0.3 if keyword else 0.0), item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:20]]
