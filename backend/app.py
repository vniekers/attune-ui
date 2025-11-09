# --- Environment loading (Render-safe; .env optional) ---
import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"

# In local dev, load .env if it exists. In Render, env vars come from the dashboard.
if ENV_PATH.exists():
    # Do NOT override already-set process envs (e.g., from Render)
    load_dotenv(dotenv_path=ENV_PATH, override=False)

# Required keys must be present either from .env (dev) or dashboard (Render)
REQUIRED = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
    "NEON_URL", "QDRANT_URL", "API_AUTH_TOKEN"
]
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
        + ". In production, set them in Render → Settings → Environment. "
        + "In local dev, add them to backend/.env"
    )

import json
import uuid
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
#from sentence_transformers import SentenceTransformer
from .embeddings import embed_texts  # or embed_texts_sync

# async context (inside FastAPI endpoints/tasks):
vectors = await embed_texts(texts)

# sync scripts / one-off ingestion:
vectors = embed_texts_sync(texts)

from openai import OpenAI


# === CONFIGURATION ===
OPENAI_BASE = os.environ["OPENAI_BASE_URL"]
OPENAI_MODEL = os.environ["OPENAI_MODEL"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]
NEON_URL = os.environ["NEON_URL"]
QDRANT_URL = os.environ["QDRANT_URL"]
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
PRIVATE_COL = os.getenv("PRIVATE_COLLECTION", "app-private")
SHARED_COL = os.getenv("SHARED_COLLECTION", "app-shared")
API_AUTH = os.getenv("API_AUTH_TOKEN", "")

# === INITIALIZE CLIENTS ===
engine = create_engine(NEON_URL, pool_pre_ping=True, future=True)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer(EMBED_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()
client = OpenAI(base_url=OPENAI_BASE, api_key=OPENAI_KEY)

# === DATABASE + VECTOR SETUP ===
def ensure_schema():
    with engine.begin() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS chats (
            id UUID PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            messages JSONB NOT NULL
        );"""))
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY,
            chat_id UUID NOT NULL,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );"""))
    names = [c.name for c in qdrant.get_collections().collections]
    for col in [PRIVATE_COL, SHARED_COL]:
        if col not in names:
            qdrant.create_collection(
                collection_name=col,
                vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
            )
        # NEW: ensure payload index for user_id
        try:
            qdrant.create_payload_index(
                collection_name=col,
                field_name="user_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # index probably exists already — ignore
            pass


ensure_schema()

app = FastAPI(title="Codex Memory API")

# === MODELS ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_id: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    use_shared_memory: Optional[bool] = False
    send_memory_to_llm: Optional[bool] = True


# === HELPERS ===
def redact(text: str) -> str:
    """Remove obvious PII before sending to LLM."""
    import re
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[redacted-email]', text)
    text = re.sub(r'\b\+?\d[\d\-\s]{7,}\b', '[redacted-phone]', text)
    return text


def embed(texts: List[str]):
    return embedder.encode(texts, normalize_embeddings=True).tolist()


def memory_upsert(user_id: str, msgs: List[ChatMessage], use_shared: bool):
    """Store user messages in Qdrant."""
    col = PRIVATE_COL
    points = []
    ts = int(time.time() * 1000)
    for i, m in enumerate(msgs):
        if m.role != "user":
            continue
        vec = embed([m.content])[0]
        points.append(qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"user_id": user_id, "role": m.role, "content": m.content, "ts": ts + i}
        ))
    if points:
        qdrant.upsert(collection_name=col, points=points)
        if use_shared:
            qdrant.upsert(collection_name=SHARED_COL, points=points)


def memory_search(user_id: str, query: str, k: int = 5):
    """Retrieve top-k similar past messages for context."""
    vec = embed([query])[0]
    out = qdrant.search(
        collection_name=PRIVATE_COL,
        query_vector=vec,
        limit=k,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]
        ),
    )
    if len(out) < k:
        out += qdrant.search(collection_name=SHARED_COL, query_vector=vec, limit=k)
    return [{"content": r.payload.get("content", "")} for r in out]

def persist_chat(user_id: str, messages: List[Dict[str, Any]]):
    """Persist chat logs into Neon DB (messages stored as JSONB)."""
    chat_id = str(uuid.uuid4())
    with engine.begin() as con:
        # store the whole conversation blob in chats.messages (JSONB)
        con.execute(
            text("""
                INSERT INTO chats (id, user_id, messages)
                VALUES (:id, :uid, (:msgs)::jsonb)
            """),
            {
                "id": chat_id,
                "uid": user_id,
                "msgs": json.dumps(messages),  # <-- serialize
            },
        )
        # also store each message row for simpler querying
        for m in messages:
            con.execute(
                text("""
                    INSERT INTO chat_messages (id, chat_id, user_id, role, content)
                    VALUES (:id, :cid, :uid, :role, :content)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "cid": chat_id,
                    "uid": user_id,
                    "role": m["role"],
                    "content": m["content"],
                },
            )
    return chat_id



# === AUTH ===
def require_auth(token: str = Header(None, convert_underscores=False, alias="Authorization")):
    if not API_AUTH:
        return
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if token.split(" ", 1)[1] != API_AUTH:
        raise HTTPException(status_code=403, detail="Invalid token")


# === ROUTES ===
@app.get("/health")
def health():
    """Simple health check for Neon, Qdrant, and Fireworks."""
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        qdrant.get_collections()
        client.models.list()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# @app.post("/chat")
# def chat(req: ChatRequest, _: None = require_auth()):
@app.post("/chat")
def chat(req: ChatRequest, _: None = Depends(require_auth)):
    """Main chat endpoint."""
    recall = memory_search(req.user_id, req.messages[-1].content, k=4)
    mem_block = "\n".join(f"- {r['content']}" for r in recall if r.get("content"))

    system_prefix = {
        "role": "system",
        "content": "You are Codex OS. Use relevant memory if helpful:\n" + mem_block
    }

    oai_messages = [system_prefix] + [m.model_dump() for m in req.messages]
    if not req.send_memory_to_llm:
        oai_messages = [system_prefix] + [
            {"role": m.role, "content": redact(m.content)} for m in req.messages
        ]

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=oai_messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    reply = completion.choices[0].message
    all_msgs = oai_messages + [reply.model_dump()]
    persist_chat(req.user_id, all_msgs)
    memory_upsert(req.user_id, req.messages, req.use_shared_memory)

    return {"reply": reply.model_dump(), "used_memory": recall}
