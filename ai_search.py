import json
import mimetypes
import os
import time
from pathlib import Path
 
import numpy as np
from PIL import Image, ImageOps
 
Image.MAX_IMAGE_PIXELS = 25_000_000  # cap decoded image memory on small instances
 
# CLIP is deliberately opt-in because the Render Free instance has only 512 MB RAM.
# Set ENABLE_CLIP=true only on a host with sufficient memory.
_CLIP_REQUESTED = os.getenv("IMAGE_EMBED_BACKEND", "visual").strip().lower() == "clip"
_CLIP_ALLOWED = os.getenv("ENABLE_CLIP", "false").strip().lower() in ("1", "true", "yes", "on")
IMAGE_BACKEND = "clip" if _CLIP_REQUESTED and _CLIP_ALLOWED else "visual"
 
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
 
 
_clip = None
_clip_text = None


def _clip_vector(path):
    """Encode an image with CLIP's vision encoder."""
    global _clip
    _open_rgb(path)  # validates the file first
    if _clip is None:
        try:
            from fastembed import ImageEmbedding
        except ImportError as exc:
            raise RuntimeError("CLIP search needs fastembed installed in requirements.txt") from exc
        _clip = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
    vector = np.asarray(next(iter(_clip.embed([str(path)]))), dtype=np.float64)
    vector = _normalize(vector, center=False)
    return {"model": "clip-vit-b32", "vec": [round(float(x), 5) for x in vector]}


def clip_text_vector(text):
    """Encode text in the same vector space as CLIP image embeddings."""
    global _clip_text
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter a search description.")
    if _clip_text is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("CLIP search needs fastembed installed in requirements.txt") from exc
        _clip_text = TextEmbedding("Qdrant/clip-ViT-B-32-text")
    vector = np.asarray(next(iter(_clip_text.embed([text[:500]]))), dtype=np.float64)
    vector = _normalize(vector, center=False)
    return {"model": "clip-vit-b32", "vec": [round(float(x), 5) for x in vector]}
 
 
def image_vector(path):
    """Embedding record {'model': ..., 'vec': [...]} for one image file."""
    return _clip_vector(path) if IMAGE_BACKEND == "clip" else _visual_vector(path)
 
 
def current_image_model():
    return "clip-vit-b32" if IMAGE_BACKEND == "clip" else "visual-v1"


def current_text_model():
    if IMAGE_BACKEND == "clip":
        return "clip-vit-b32"
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
    """Rank photos against text with CLIP; retain keyword fallback for other media."""
    query = (query or "").strip()
    if not query:
        return []

    words = [w for w in query.casefold().split() if len(w) > 2]
    scored = []

    if IMAGE_BACKEND == "clip":
        query_record = clip_text_vector(query)
        for row in rows:
            text = f"{row['filename']} {row['description']}".casefold()
            keyword = bool(words) and any(w in text for w in words)
            entry = load_record(row["embedding"]).get("img")
            semantic = 0.0
            if row["media_type"] == "Image" and entry and entry.get("model") == query_record["model"] and entry.get("vec"):
                semantic = cosine_similarity(query_record["vec"], entry["vec"])
            # CLIP cosine scores are ranking signals, not calibrated probabilities.
            if (row["media_type"] == "Image" and semantic > 0.10) or keyword:
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
