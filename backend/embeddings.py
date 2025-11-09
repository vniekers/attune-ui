# backend/embeddings.py
import os
import httpx
from typing import List

FIREWORKS_API_KEY = os.environ["OPENAI_API_KEY"]
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")  # choose your remote model

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Async embedding call (use inside FastAPI routes or async tasks)."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"{BASE}/embeddings",
            headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
            json={"model": MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in data]

def embed_texts_sync(texts: List[str]) -> List[List[float]]:
    """Sync wrapper (use in scripts or sync code paths)."""
    with httpx.Client(timeout=15) as c:
        r = c.post(
            f"{BASE}/embeddings",
            headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
            json={"model": MODEL, "input": texts},
        )
        r.raise_for_status()
        data = r.json()["data"]
        return [d["embedding"] for d in data]
