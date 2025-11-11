# backend/vector_upsert.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import os, time, hashlib, uuid
from typing import List, Dict, Any

QDRANT_URL  = os.environ["QDRANT_URL"]
QDRANT_KEY  = os.environ.get("QDRANT_API_KEY")
COLLECTION  = os.environ.get("PRIVATE_COLLECTION", "codex-private")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def ensure_collection(dim: int, distance: str = "Cosine"):
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=getattr(qm.Distance, distance.upper())),
    )

def upsert_texts(ids: List[str], texts: List[str], vectors: List[List[float]], meta: Dict[str, Any] | None = None):
    ts = int(time.time())
    points = []
    for _id, t, v in zip(ids, texts, vectors):
        payload = {"model": EMBED_MODEL, "text_sha1": sha1(t), "created_at": ts}
        if meta:
            payload.update(meta)
        points.append(qm.PointStruct(id=str(_id) or str(uuid.uuid4()), vector=v, payload=payload))
    client.upsert(collection_name=COLLECTION, points=points, wait=True)
