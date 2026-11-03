import json
import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# Several vision models are tried in order, so one retired name does not break the app.
VISION_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash").split(",")
    if m.strip()
]
EMBED_DIMS = 768
MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.55"))
TIMEOUT_MS = int(os.getenv("AI_TIMEOUT_MS", "20000"))

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=TIMEOUT_MS),
        )
    return _client


def make_embedding(text):
    text = (text or "").strip()
    if not text:
        return None
    result = _get_client().models.embed_content(
        model=EMBED_MODEL,
        contents=text[:4000],
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIMS),
    )
    return list(result.embeddings[0].values)


def describe_image(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
    prompt = "Describe this image in 2-3 plain sentences: main objects, colors, setting and actions."
    errors = []
    for model in VISION_MODELS:
        try:
            response = _get_client().models.generate_content(model=model, contents=[image_part, prompt])
            text = (response.text or "").strip()
            if text:
                return text
            errors.append(f"{model}: empty response")
        except Exception as exc:
            errors.append(f"{model}: {exc!r}"[:300])
    raise RuntimeError("All vision models failed: " + " | ".join(errors))


def describe_and_embed(path):
    description = describe_image(path)
    return description, make_embedding(description)


def make_image_embedding(path):
    return describe_and_embed(path)[1]


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_hit(query, row):
    words = [w for w in query.casefold().split() if len(w) > 2]
    text = f"{row['filename']} {row['description']}".casefold()
    return bool(words) and any(w in text for w in words)


def search_media(query, rows):
    try:
        query_vector = make_embedding(query)
    except Exception as exc:
        print(f"Query embedding failed, using keyword search: {exc!r}", flush=True)
        query_vector = None

    scored = []
    for row in rows:
        semantic = 0.0
        if query_vector:
            try:
                vector = json.loads(row["embedding"]) if row["embedding"] else None
            except (TypeError, ValueError):
                vector = None
            semantic = cosine_similarity(query_vector, vector)
        keyword = _keyword_hit(query, row)
        if semantic >= MIN_SCORE or keyword:
            scored.append((semantic + (0.3 if keyword else 0.0), row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:20]]
