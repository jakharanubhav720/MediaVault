import json
import mimetypes
import os
from pathlib import Path

from google import genai
from google.genai import types

EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
EMBED_DIMS = 768
MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.5"))

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
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
    response = _get_client().models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=path.read_bytes(), mime_type=mime),
            "Describe this image in 2-3 plain sentences: main objects, colors, setting and actions.",
        ],
    )
    return (response.text or "").strip()


def make_image_embedding(path):
    return make_embedding(describe_image(path))


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def search_media(query, rows):
    query_vector = make_embedding(query)
    if not query_vector:
        return []
    scored = []
    for row in rows:
        try:
            vector = json.loads(row["embedding"]) if row["embedding"] else None
        except (TypeError, ValueError):
            vector = None
        score = cosine_similarity(query_vector, vector)
        if score >= MIN_SCORE:
            scored.append((score, row))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:20]]
