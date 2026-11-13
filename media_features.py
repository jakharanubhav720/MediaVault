"""MediaVault organization features: local AI tags, duplicate fingerprints and smart albums."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

# Tags are intentionally broad and conservative. SigLIP 2 compares each label
# in the same shared image/text embedding space used by MediaVault search.
ALBUM_LABELS = {
    "Food": ["food", "meal", "fruit", "dessert", "drink", "cooking"],
    "Travel": ["travel", "city", "street", "landmark", "building", "transport"],
    "Nature": ["nature", "mountain", "forest", "beach", "flower", "wildlife"],
    "People": ["person", "people", "portrait", "group of people"],
    "Pets & Animals": ["dog", "cat", "bird", "animal", "wildlife"],
    "Vehicles": ["car", "motorcycle", "bus", "train", "airplane", "boat"],
    "Documents": ["document", "paper", "receipt", "book", "screenshot", "text"],
    "Architecture": ["architecture", "building", "house", "interior", "bridge"],
    "Events": ["celebration", "party", "concert", "wedding", "sports event"],
    "Technology": ["computer", "phone", "electronic device", "technology"],
}
_TAG_CACHE = {}

def image_fingerprints(path):
    """Return a cryptographic hash and 64-bit difference hash for an image."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return digest, f"{bits:016x}"

def _hamming(left, right):
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 64

def tag_image(path):
    """Generate conservative category tags using local SigLIP 2; never blocks upload on failure."""
    import ai_search
    if ai_search.IMAGE_BACKEND != "siglip2":
        return []
    image = ai_search.image_vector(path)
    tags = []
    for album, labels in ALBUM_LABELS.items():
        scores = []
        for label in labels:
            prompt = f"a photo of {label}"
            if prompt not in _TAG_CACHE:
                _TAG_CACHE[prompt] = ai_search.siglip2_text_vector(prompt)["vec"]
            scores.append((ai_search.cosine_similarity(image["vec"], _TAG_CACHE[prompt]), label))
        score, label = max(scores, default=(0.0, ""))
        # A fixed conservative floor avoids assigning a category to every image.
        if score >= 0.12:
            tags.append({"tag": label, "album": album, "score": round(float(score), 4)})
    return tags

def duplicate_groups(rows, upload_dir, near_threshold=6):
    """Return exact and near-duplicate groups. No files are deleted automatically."""
    valid = []
    for row in rows:
        if row["media_type"] != "Image":
            continue
        path = Path(upload_dir) / row["stored_filename"]
        if not path.is_file():
            continue
        digest = row["sha256"] if "sha256" in row.keys() else None
        phash = row["phash"] if "phash" in row.keys() else None
        if not digest or not phash:
            try:
                digest, phash = image_fingerprints(path)
            except Exception:
                continue
        valid.append((dict(row), digest, phash))
    groups = []
    used = set()
    # Exact byte-for-byte duplicates first.
    exact = defaultdict(list)
    for item, digest, phash in valid:
        exact[digest].append((item, phash))
    for entries in exact.values():
        if len(entries) > 1:
            ids = {entry[0]["id"] for entry in entries}
            used |= ids
            groups.append({"kind": "Exact duplicates", "items": [entry[0] for entry in entries], "distance": 0})
    # Then visually similar images, using a perceptual difference hash.
    for i, (item, digest, phash) in enumerate(valid):
        if item["id"] in used:
            continue
        cluster = [item]
        distances = []
        for other, other_digest, other_hash in valid[i + 1:]:
            if other["id"] in used or digest == other_digest:
                continue
            distance = _hamming(phash, other_hash)
            if distance <= near_threshold:
                cluster.append(other)
                distances.append(distance)
        if len(cluster) > 1:
            ids = {entry["id"] for entry in cluster}
            used |= ids
            groups.append({"kind": "Visually similar", "items": cluster, "distance": min(distances) if distances else 0})
    return groups
