# main.py
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
import os
import time
import json
import math
import sqlite3
import secrets
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, Query, HTTPException, Request, Form
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

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

ADMIN_TOKEN_SECRET = os.getenv("ADMIN_TOKEN_SECRET", "").strip() or ("CHANGE_ME_" + secrets.token_urlsafe(16))
serializer = URLSafeSerializer(ADMIN_TOKEN_SECRET, salt="booking-approval")

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # e.g. https://freedive-chat-backend.onrender.com

# Your website forms (change these to your real URLs)
FORM_EN_URL = os.getenv("FORM_EN_URL", "").strip() or "https://YOURDOMAIN.COM/form1.php"
FORM_AR_URL = os.getenv("FORM_AR_URL", "").strip() or "https://YOURDOMAIN.COM/form1ar.php"
FORM_DE_URL = os.getenv("FORM_DE_URL", "").strip() or "https://YOURDOMAIN.COM/form1de.php"

DB_PATH = os.getenv("DB_PATH", "data.sqlite3")
MAX_URLS = int(os.getenv("MAX_URLS", "150"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "800"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.20"))

# Admin dashboard protection (simple)
ADMIN_DASH_TOKEN = os.getenv("ADMIN_DASH_TOKEN", "").strip()  # set this in Render env!


# =========================
# OpenAI (async)
# =========================
openai_client = None
if OPENAI_API_KEY:
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def _require_openai():
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")


# =========================
# Helpers
# =========================
def now_ts() -> int:
    return int(time.time())


def escape_html(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return (dot / denom) if denom else 0.0


def detect_language(text: str, accept_language: str = "") -> str:
    """
    Returns: 'ar' | 'de' | 'en'
    Uses a safe heuristic (no extra dependencies).
    """
    t = (text or "").strip()

    # Strong signal: Arabic characters
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", t):
        return "ar"

    # Accept-Language header hint
    al = (accept_language or "").lower()
    if "ar" in al:
        return "ar"
    if "de" in al:
        return "de"
    if "en" in al:
        return "en"

    # German hints
    if re.search(r"[äöüßÄÖÜ]", t) or re.search(r"\b(und|oder|bitte|preise|kurs|tauchen|freitauchen)\b", t.lower()):
        return "de"

    return "en"


def booking_form_url(lang: str) -> str:
    if lang == "ar":
        return FORM_AR_URL
    if lang == "de":
        return FORM_DE_URL
    return FORM_EN_URL


# =========================
# Email (SendGrid)
# =========================
def send_email(to_email: str, subject: str, html: str):
    if not SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY not set")
    if not to_email:
        raise RuntimeError("OWNER_NOTIFY_EMAIL not set")

    url = "https://api.sendgrid.com/v3/mail/send"
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": (os.getenv("SMTP_FROM", "").strip() or to_email)},
        "subject": subject,
        "content": [{"type": "text/html", "value": html}],
    }
    headers = {"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid error {r.status_code}: {r.text}")


def safe_send_owner_email(subject: str, html: str):
    if not (OWNER_NOTIFY_EMAIL and SENDGRID_API_KEY):
        return
    try:
        send_email(OWNER_NOTIFY_EMAIL, subject, html)
    except Exception:
        pass


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

    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        title TEXT,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL
      )
    """
    )

    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL
      )
    """
    )

    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS human_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        text TEXT NOT NULL,
        ts INTEGER NOT NULL
      )
    """
    )

    # booking status: pending -> approved/denied -> completed
    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS booking_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        status TEXT NOT NULL,
        details_json TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    """
    )

    conn.commit()
    conn.close()


def ensure_session(session_id: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO sessions(session_id, created_at) VALUES(?,?)", (session_id, now_ts()))
    conn.commit()
    conn.close()


def add_human_message(session_id: str, text: str):
    conn = db()
    cur = conn.cursor()
    cur.execute("INSERT INTO human_messages(session_id, text, ts) VALUES(?,?,?)", (session_id, text, now_ts()))
    conn.commit()
    conn.close()


def get_human_messages(session_id: str) -> List[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT text, ts FROM human_messages WHERE session_id=? ORDER BY ts ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": "Abood Freediver Team", "text": r["text"], "ts": r["ts"]} for r in rows]


def chunks_count() -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM chunks")
    n = cur.fetchone()["n"]
    conn.close()
    return int(n)


# =========================
# OpenAI helpers
# =========================
async def embed_text(text: str) -> List[float]:
    _require_openai()
    resp = await openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text[:6000])
    return list(resp.data[0].embedding)


async def generate_answer(system: str, user: str) -> str:
    _require_openai()
    resp = await openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


# =========================
# Retrieval
# =========================
async def retrieve_site_context(question: str) -> Tuple[List[Dict[str, Any]], float]:
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
    confidence = float(top[0][0]) if top else 0.0
    chunks = [{"score": s, "url": u, "title": (t or ""), "text": tx} for (s, u, t, tx) in top]
    return chunks, confidence


# =========================
# SerpAPI (safe)
# =========================
def serpapi_search(query: str) -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": 5},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for item in data.get("organic_results", [])[:5]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return out
    except Exception:
        return []


# =========================
# Booking flow
# =========================
def looks_like_booking(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        "book",
        "booking",
        "reserve",
        "reservation",
        "schedule",
        "fun dive",
        "course",
        "training",
        "session",
        "availability",
        "price",
        "join",
        "padi",
    ]
    return any(t in ql for t in triggers)


def create_booking_request(session_id: str, details: Dict[str, Any]) -> int:
    conn = db()
    cur = conn.cursor()
    ts = now_ts()
    cur.execute(
        "INSERT INTO booking_requests(session_id,status,details_json,created_at,updated_at) VALUES(?,?,?,?,?)",
        (session_id, "pending", json.dumps(details), ts, ts),
    )
    conn.commit()
    booking_id = int(cur.lastrowid)
    conn.close()
    return booking_id


def update_booking(booking_id: int, status: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM booking_requests WHERE id=?", (booking_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")

    new_status = status or row["status"]
    current_details = json.loads(row["details_json"] or "{}")
    if details:
        current_details.update(details)

    cur.execute(
        "UPDATE booking_requests SET status=?, details_json=?, updated_at=? WHERE id=?",
        (new_status, json.dumps(current_details), now_ts(), booking_id),
    )
    conn.commit()
    conn.close()


def get_booking(booking_id: int) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM booking_requests WHERE id=?", (booking_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    d = dict(row)
    try:
        d["details"] = json.loads(d.get("details_json") or "{}")
    except Exception:
        d["details"] = {}
    return d


def get_latest_booking_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM booking_requests WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d["details"] = json.loads(d.get("details_json") or "{}")
    except Exception:
        d["details"] = {}
    return d


def make_approval_links(base_url: str, booking_id: int) -> Tuple[str, str]:
    token = serializer.dumps({"booking_id": booking_id})
    approve = f"{base_url}/admin/booking/approve?token={token}"
    deny = f"{base_url}/admin/booking/deny?token={token}"
    return approve, deny


# =========================
# Admin auth
# =========================
def require_admin(request: Request):
    if not ADMIN_DASH_TOKEN:
        raise HTTPException(status_code=500, detail="ADMIN_DASH_TOKEN not set")
    given = request.headers.get("x-admin-token", "").strip()
    if not given:
        given = request.query_params.get("token", "").strip()
    if given != ADMIN_DASH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


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
    history: Optional[List[Dict[str, Any]]] = None
    page_url: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    needs_human: bool = False
    booking_pending: bool = False
    # Keep field for compatibility, but frontend can ignore.
    sources: Optional[List[Dict[str, str]]] = None


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/chat/status")
def chat_status(session_id: str = Query(...)):
    return {"messages": get_human_messages(session_id)}


# -------- Booking submit webhook (your form should POST here) --------
# This is how you "auto-close booking after form submit".
# Your PHP form can POST to: https://freedive-chat-backend.onrender.com/booking/submit
@app.post("/booking/submit")
async def booking_submit(
    request: Request,
    booking_id: int = Form(...),
    full_name: str = Form(""),
    age: str = Form(""),
    whatsapp_or_phone: str = Form(""),
    email: str = Form(""),
    session_type: str = Form(""),
    preferred_date_time: str = Form(""),
    people_count: str = Form(""),
    certified_status: str = Form(""),
    height_cm: str = Form(""),
    weight_kg: str = Form(""),
    feet_size: str = Form(""),
    need_equipment: str = Form(""),
    underwater_photography: str = Form(""),
    payment_method: str = Form(""),
    preferred_dive_site: str = Form(""),
    training_focus: str = Form(""),
):
    # If you want to protect this endpoint, set BOOKING_SUBMIT_TOKEN and check header.
    # For now, keep it open because it comes from your website form.
    booking = get_booking(booking_id)
    details = booking.get("details") or {}

    customer = details.get("customer_info") or {}
    customer.update(
        {
            "full_name": full_name.strip(),
            "age": age.strip(),
            "whatsapp_or_phone": whatsapp_or_phone.strip(),
            "email": email.strip(),
            "session_type": session_type.strip(),
            "date_time": preferred_date_time.strip(),
            "people_count": people_count.strip(),
            "certification_status": certified_status.strip(),
            "height_cm": height_cm.strip(),
            "weight_kg": weight_kg.strip(),
            "feet_size": feet_size.strip(),
            "need_equipment": need_equipment.strip(),
            "underwater_photography": underwater_photography.strip(),
            "payment_method": payment_method.strip(),
            "preferred_dive_site": preferred_dive_site.strip(),
            "training_focus": training_focus.strip(),
        }
    )

    details["customer_info"] = customer
    details.setdefault("form_submissions", [])
    details["form_submissions"].append({"ts": now_ts(), "ip": request.client.host if request.client else ""})

    # Auto-close booking
    update_booking(booking_id, status="completed", details=details)

    # Notify owner with all details
    safe_send_owner_email(
        "Booking form submitted (auto-closed)",
        f"""
        <h3>Booking completed</h3>
        <p><b>Booking ID:</b> {booking_id}</p>
        <p><b>Session:</b> {escape_html(booking.get("session_id",""))}</p>
        <p><b>Customer info:</b></p>
        <pre>{escape_html(json.dumps(customer, indent=2))}</pre>
        """,
    )

    # Optional: push a human message to the chat
    add_human_message(
        booking["session_id"],
        "Abood Freediver Team: Thanks — we received your booking details. We will confirm the final schedule and meeting point shortly.",
    )

    return {"ok": True, "status": "completed", "booking_id": booking_id}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    if not req.question or not req.session_id:
        raise HTTPException(status_code=400, detail="question and session_id required")

    ensure_session(req.session_id)
    q = (req.question or "").strip()

    lang = detect_language(q, accept_language=request.headers.get("accept-language", ""))
    form_url = booking_form_url(lang)

    # If booking already approved: provide clickable form link (no complex template in chat)
    latest = get_latest_booking_for_session(req.session_id)
    if latest and latest["status"] == "approved":
        booking_id = int(latest["id"])
        link = f"{form_url}?booking_id={booking_id}&session_id={req.session_id}"

        if lang == "ar":
            msg = (
                "فريق عبود فريدايفر: تمّت الموافقة على طلب الحجز.\n\n"
                f"لإكمال الحجز، رجاءً افتح الرابط واملأ النموذج:\n{link}\n\n"
                "إذا لديك أي سؤال آخر، اكتب هنا وسنساعدك."
            )
        elif lang == "de":
            msg = (
                "Abood Freediver Team: Deine Buchung wurde genehmigt.\n\n"
                f"Bitte öffne den Link und fülle das Formular aus:\n{link}\n\n"
                "Wenn du noch Fragen hast, schreibe hier weiter."
            )
        else:
            msg = (
                "Abood Freediver Team: Your booking request is approved.\n\n"
                f"Please open this link and complete the form:\n{link}\n\n"
                "If you have any other questions, you can continue chatting here."
            )

        # notify owner that user was sent to form
        safe_send_owner_email(
            "Booking approved: form link sent",
            f"""
            <h3>Form link sent to customer</h3>
            <p><b>Booking ID:</b> {booking_id}</p>
            <p><b>Session:</b> {escape_html(req.session_id)}</p>
            <p><b>Language:</b> {escape_html(lang)}</p>
            <p><b>Form link:</b> <a href="{escape_html(link)}">{escape_html(link)}</a></p>
            """,
        )

        return ChatResponse(answer=msg, needs_human=True, booking_pending=True, sources=[])

    # Booking intent -> create pending booking, notify owner, tell user "pending"
    if looks_like_booking(q):
        booking_details = {
            "question": q,
            "page_url": req.page_url or "",
            "history": req.history or [],
            "language": lang,
            "raw_messages": [{"ts": now_ts(), "text": q}],
        }
        booking_id = create_booking_request(req.session_id, booking_details)

        approve, deny = ("", "")
        if PUBLIC_BASE_URL:
            approve, deny = make_approval_links(PUBLIC_BASE_URL, booking_id)

        safe_send_owner_email(
            "Booking request pending approval",
            f"""
            <h3>Booking request pending approval</h3>
            <p><b>Booking ID:</b> {booking_id}</p>
            <p><b>Session:</b> {escape_html(req.session_id)}</p>
            <p><b>Page:</b> {escape_html(req.page_url or "")}</p>
            <p><b>User message:</b></p>
            <pre>{escape_html(q)}</pre>
            <p><b>Language:</b> {escape_html(lang)}</p>
            <p><b>Approve:</b> <a href="{approve}">{approve}</a></p>
            <p><b>Deny:</b> <a href="{deny}">{deny}</a></p>
            """,
        )

        if lang == "ar":
            msg = (
                "فريق عبود فريدايفر: استلمنا طلب الحجز.\n"
                "لا يمكننا تأكيد الحجز إلا بعد موافقة المدرب.\n\n"
                "سوف نرسل لك رابط نموذج التفاصيل بعد الموافقة."
            )
        elif lang == "de":
            msg = (
                "Abood Freediver Team: Wir haben deine Buchungsanfrage erhalten.\n"
                "Wir können erst nach Freigabe durch den Instructor bestätigen.\n\n"
                "Nach der Freigabe schicken wir dir einen Link zum Formular."
            )
        else:
            msg = (
                "Abood Freediver Team: We received your booking request.\n"
                "We can only confirm after instructor approval.\n\n"
                "After approval, we will send you a link to a short form to collect details."
            )

        return ChatResponse(answer=msg, needs_human=True, booking_pending=True, sources=[])

    # Normal Q&A (RAG)
    chunks, conf = await retrieve_site_context(q)

    system = (
        "You are Aqua from the Abood Freediver Team (Aqaba).\n"
        "Always speak as the Abood Freediver Team.\n"
        "Rules:\n"
        "- Prefer answering using SITE_CONTEXT.\n"
        "- If SITE_CONTEXT is insufficient, you may use WEB_CONTEXT if provided.\n"
        "- Never confirm a booking unless it is approved.\n"
        "- Keep answers concise, accurate, and safety-conscious.\n"
        "- Do not show sources unless the user explicitly asks for sources.\n"
        "- Do not mention internal implementation.\n"
    )

    site_context = ""
    for c in chunks:
        site_context += f"\n\nSOURCE: {c['url']}\n{c['text'][:2000]}"

    web_results = serpapi_search(q) if conf < MIN_CONFIDENCE else []
    web_context = ""
    for r in web_results:
        web_context += f"\n\nWEB: {r.get('title','')}\n{r.get('link','')}\n{r.get('snippet','')}"

    prompt = (
        f"USER_QUESTION:\n{q}\n\n"
        f"SITE_CONTEXT:\n{site_context if site_context else '(none)'}\n\n"
        f"WEB_CONTEXT:\n{web_context if web_context else '(none)'}\n\n"
        "TASK:\n"
        "- Answer the user question.\n"
        "- If information is missing, ask 1 short follow-up question.\n"
        "- Do not include a Sources section.\n"
    )

    answer = await generate_answer(system, prompt)
    return ChatResponse(answer=answer, needs_human=False, booking_pending=False, sources=[])


# =========================
# Admin endpoints (dashboard backend)
# =========================
@app.get("/admin/bookings")
def admin_list_bookings(request: Request, status: str = Query(default="")):
    require_admin(request)
    conn = db()
    cur = conn.cursor()
    if status:
        rows = cur.execute(
            "SELECT id, session_id, status, created_at, updated_at FROM booking_requests WHERE status=? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT id, session_id, status, created_at, updated_at FROM booking_requests ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return {"bookings": [dict(r) for r in rows]}


@app.get("/admin/bookings/{booking_id}")
def admin_get_booking(booking_id: int, request: Request):
    require_admin(request)
    return get_booking(booking_id)


# =========================
# Approval links
# =========================
@app.get("/admin/booking/approve")
def approve_booking(token: str = Query(...)):
    try:
        data = serializer.loads(token)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    booking_id = int(data["booking_id"])
    booking = get_booking(booking_id)

    update_booking(booking_id, status="approved")

    # Send a human message that will appear in the widget (optional)
    add_human_message(
        booking["session_id"],
        "Abood Freediver Team: Approved. Please continue by completing the booking form link we will provide in chat.",
    )

    safe_send_owner_email(
        "Booking approved",
        f"""
        <h3>Booking approved</h3>
        <p><b>Booking ID:</b> {booking_id}</p>
        <p><b>Session:</b> {escape_html(booking.get("session_id",""))}</p>
        <p>The customer will receive the form link in chat when they message next.</p>
        """,
    )

    return {"ok": True, "status": "approved", "booking_id": booking_id}


@app.get("/admin/booking/deny")
def deny_booking(token: str = Query(...)):
    try:
        data = serializer.loads(token)
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid token")

    booking_id = int(data["booking_id"])
    booking = get_booking(booking_id)

    update_booking(booking_id, status="denied")

    add_human_message(
        booking["session_id"],
        "Abood Freediver Team: We can’t confirm that booking request yet. Please share alternative dates/times and your experience level.",
    )

    safe_send_owner_email(
        "Booking denied",
        f"""
        <h3>Booking denied</h3>
        <p><b>Booking ID:</b> {booking_id}</p>
        <p><b>Session:</b> {escape_html(booking.get("session_id",""))}</p>
        """,
    )

    return {"ok": True, "status": "denied", "booking_id": booking_id}
