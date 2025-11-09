# backend/vector_upsert.py
import time
import hashlib
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import os

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION = os.environ.get("PRIVATE_COLLECTION", "codex-private")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def upsert_texts(
    ids: List[str],
    texts: List[str],
    vectors: List[List[float]],
    meta: Dict[str, Any] | None = None,
):
    """Insert or update texts + vectors into Qdrant with metadata."""
    ts = int(time.time())
    payloads = []
    for _id, t in zip(ids, texts):
        pl = {
            "text_sha1": _sha1(t),
            "model": EMBED_MODEL,
            "created_at": ts,
        }
        if meta:
            pl.update(meta)
        payloads.append(pl)

    client.upsert(
        collection_name=COLLECTION,
        points=qmodels.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        ),
        wait=True,
    )
