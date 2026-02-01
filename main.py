from pathlib import Path
from dotenv import load_dotenv
import os
import time
import json
import math
import sqlite3
import secrets
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bs4 import BeautifulSoup
from itsdangerous import URLSafeSerializer, BadSignature

# =========================
# Load .env (local only)
# =========================
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(env_path)

# =========================
# Config
# =========================
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL", "").strip()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SEARCHAPI_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
] or ["*"]

ADMIN_TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET") or "CHANGE_ME_" + secrets.token_urlsafe(16)
serializer = URLSafeSerializer(ADMIN_TOKEN_SECRET, salt="booking-approval")

DB_PATH = os.getenv("DB_PATH", "data.sqlite3")

MAX_URLS = int(os.getenv("MAX_URLS", "150"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "800"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))

# =========================
# OpenAI (async, modern)
# =========================
from openai import AsyncOpenAI
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =========================
# Database
# =========================
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT,
            text TEXT,
            embedding TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS human_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            text TEXT,
            ts INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS booking_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            status TEXT,
            details_json TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
    """)
    conn.commit()
    conn.close()


def chunks_count() -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    n = cur.fetchone()[0]
    conn.close()
    return n


# =========================
# Utilities
# =========================
def now_ts() -> int:
    return int(time.time())


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def strip_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    text = "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())
    return title, text


def chunk_text(text: str) -> List[str]:
    chunks = []
    i = 0
    while i < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(len(text), i + CHUNK_CHARS)
        chunks.append(text[i:end])
        i = end - CHUNK_OVERLAP
    return chunks


# =========================
# OpenAI helpers
# =========================
async def embed_text(text: str) -> List[float]:
    resp = await openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text[:6000],
    )
    return resp.data[0].embedding


async def generate_answer(system: str, user: str) -> str:
    resp = await openai_client.responses.create(
        model=OPENAI_CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.output_text.strip()


# =========================
# Retrieval (FIXED)
# =========================
async def retrieve_site_context(
    question: str
) -> Tuple[List[Dict[str, Any]], float]:

    if chunks_count() == 0:
        return [], 0.0

    q_emb = await embed_text(question)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT url,title,text,embedding FROM chunks")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for r in rows:
        try:
            emb = json.loads(r["embedding"])
            score = cosine(q_emb, emb)
            scored.append((score, r["url"], r["title"], r["text"]))
        except Exception:
            continue

    if not scored:
        return [], 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    confidence = top[0][0]

    chunks = [
        {"score": s, "url": u, "title": t or "", "text": tx}
        for s, u, t, tx in top
    ]
    return chunks, confidence


# =========================
# SerpAPI (SAFE)
# =========================
def serpapi_search(query: str) -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": SERPAPI_KEY},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        return [
            {
                "title": i.get("title", ""),
                "link": i.get("link", ""),
                "snippet": i.get("snippet", ""),
            }
            for i in data.get("organic_results", [])[:5]
        ]
    except Exception:
        return []


# =========================
# FastAPI
# =========================
app = FastAPI(title="Freedive Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str
    history: Optional[list] = None
    page_url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    needs_human: bool = False
    booking_pending: bool = False
    sources: Optional[list] = None


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    chunks, conf = await retrieve_site_context(req.question)

    system = (
        "You are Aqua, a freediving assistant.\n"
        "Use SITE_CONTEXT when available.\n"
        "Be concise and safety-conscious."
    )

    site_context = ""
    sources = []

    for c in chunks:
        site_context += f"\nSOURCE: {c['url']}\n{c['text'][:1500]}"
        sources.append({"title": c["title"], "url": c["url"]})

    web_context = ""
    if conf < MIN_CONFIDENCE:
        for r in serpapi_search(req.question):
            web_context += f"\nWEB: {r['link']}\n{r['snippet']}"

    prompt = f"""
USER_QUESTION:
{req.question}

SITE_CONTEXT:
{site_context or "(none)"}

WEB_CONTEXT:
{web_context or "(none)"}
"""

    answer = await generate_answer(system, prompt)

    return ChatResponse(
        answer=answer,
        sources=sources[:4],
    )
