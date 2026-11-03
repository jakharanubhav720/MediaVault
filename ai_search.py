import json
import math
from pathlib import Path

from PIL import Image

_model = None


def get_model():
    global _model

    if _model is None:
        # Import the heavy library only when AI search is actually requested.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("clip-ViT-B-32")

    return _model


def describe_image(path: Path, mime_type: str) -> str:
    return ""


def make_embedding(text: str):
    if not text or not text.strip():
        return None

    vector = get_model().encode(
        text.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vector.tolist()


def make_image_embedding(path: Path):
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")

            vector = get_model().encode(
                image,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        return vector.tolist()

    except Exception as error:
        print(f"Could not process image {path}: {error}")
        return None


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def search_media(query: str, rows):
    query_vector = make_embedding(query)

    if query_vector is None:
        return []

    ranked = []

    for row in rows:
        try:
            vector = json.loads(row["embedding"]) if row["embedding"] else None
        except (TypeError, json.JSONDecodeError):
            vector = None

        score = cosine_similarity(query_vector, vector)

        if score > 0:
            item = dict(row)
            item["similarity"] = round(score, 4)
            ranked.append(item)

    ranked.sort(key=lambda item: item["similarity"], reverse=True)

    return ranked
