import base64
import mimetypes
import os
from pathlib import Path

from openai import OpenAI

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.25"))

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30)
    return _client


def make_embedding(text):
    text = (text or "").strip()
    if not text:
        return None
    result = _get_client().embeddings.create(model=EMBED_MODEL, input=text[:4000])
    return result.data[0].embedding


def describe_image(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    response = _get_client().chat.completions.create(
        model=VISION_MODEL,
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in 2-3 plain sentences: main objects, colors, setting and actions."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
    )
    return (response.choices[0].message.content or "").strip()


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
    import json
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
