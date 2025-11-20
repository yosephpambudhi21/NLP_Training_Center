# ===============================
# 🤖 TLC Training Chatbot — All-in-One (Colab, 1 cell)
# ===============================
import sys, subprocess, importlib, os, re, textwrap
from datetime import datetime

# ---------- 0) Install minimal deps (format Python) ----------
pkgs = [
    "gradio==4.44.1",
    "sentence-transformers==3.0.1",
    "scikit-learn==1.6.1",
    "pandas==2.2.2",
    "langdetect==1.0.9",
    "python-dateutil==2.9.0.post0",
    "posthog==3.7.2",
]
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=False)

# ---------- 1) Imports ----------
import gradio as gr
import pandas as pd
import numpy as np
from langdetect import detect
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

# ---------- 2) Config ----------
ORG_NAME = "Toyota Learning Center (TLC) – TMMIN"
TZ = "Asia/Jakarta"
CONTACT_EMAIL = "tlc@toyota.co.id"
REG_FORM_URL = "https://forms.gle/your-form-id"  # ganti dengan form kamu
WHATSAPP_LINK = "https://wa.me/6281234567890?text=Halo%20TLC%2C%20saya%20ingin%20daftar%20training"
LOCATIONS = ["TLC Sunter 2 – Zenix 2", "TLC Karawang", "Online (MS Teams)", "On-site (Supplier)"]

# Intent routing knobs — tweak to adjust how strict each layer is
INTENT_CONFIDENCE_THRESHOLD = 0.5  # classifier must be at least this confident
SEMANTIC_INTENT_THRESHOLD = 0.6    # semantic similarity must exceed this when classifier is uncertain

# Conversational context helpers (keywords for lightweight rules)
PRICING_KEYWORDS = ["harga", "biaya", "cost", "fee", "tarif"]
EXTERNAL_KEYWORDS = [
    "di luar pabrik",
    "di luar plant",
    "di luar site",
    "diadakan di luar",
    "in-house",
    "in house",
    "inhouse",
    "onsite",
    "on-site",
    "di lokasi kami",
    "bisa external",
    "bisa diadakan di luar",
]

# ---------- 3) Sample data (bisa ganti ke CSV internal) ----------
courses = pd.DataFrame([
    {
        "code": "JKK-SV-101",
        "title": "Jikotei Kanketsu for Supervisor",
        "audience": "Supervisor / Section Head (Supplier & Internal)",
        "format": "Offline",
        "duration_days": 1,
        "next_runs": "2025-11-17;2025-12-03",
        "location": "TLC Sunter 2 – Zenix 2",
        "price_idr": 1850000,
        "prereq": "Dasar PDCA",
        "description": "Prinsip JKK, PDCA, standard work audit, problem finding di lini produksi supplier."
    },
    {
        "code": "TCLASS-FOUND-201",
        "title": "TClass LMS Foundation",
        "audience": "Admin HRD Supplier, Training PIC",
        "format": "Online",
        "duration_days": 0.5,
        "next_runs": "2025-11-20;2025-12-10",
        "location": "Online (MS Teams)",
        "price_idr": 950000,
        "prereq": "-",
        "description": "Pengantar TClass: membuat kelas, import peserta, tracking jam training, laporan compliance."
    },
    {
        "code": "QCC-LEAD-301",
        "title": "QCC Leadership & Facilitation",
        "audience": "Team Leader / Champion QCC",
        "format": "Offline",
        "duration_days": 1,
        "next_runs": "2025-12-05",
        "location": "TLC Karawang",
        "price_idr": 2250000,
        "prereq": "Basic Quality Tools",
        "description": "Memandu tim QCC, fishbone, 7 tools, storytelling improvement, persiapan presentasi juri."
    },
])
faqs = pd.DataFrame([
    {"q": "Apakah program TLC terbuka untuk supplier?",
     "a": "Ya, program tertentu dibuka untuk supplier. Lihat katalog & jadwal atau hubungi kami untuk batch khusus."},
    {"q": "Metode pembayaran?",
     "a": "Transfer bank (invoice), e-invoice vendor terdaftar. Mohon PO atau konfirmasi email ke " + CONTACT_EMAIL + "."},
    {"q": "Lokasi training offline?",
     "a": f"Utamanya {LOCATIONS[0]} dan {LOCATIONS[1]}. Kami juga bisa on-site di pabrik supplier."},
    {"q": "Kebijakan pembatalan?",
     "a": "Pembatalan ≤ D-5: refund penuh; D-4 s.d D-2: 50%; D-1/no-show: tidak refund (dapat reschedule jika ada slot)."},
    {"q": "Bahasa yang didukung chatbot?",
     "a": "Bahasa Indonesia & Inggris."},
    {"q": "Apakah TLC bisa in-house di lokasi perusahaan?",
     "a": "Bisa, kami dapat mengadakan pelatihan on-site/in-house di pabrik atau kantor Anda sesuai jadwal yang disepakati."},
    {"q": "Berapa minimal peserta untuk in-house training?",
     "a": "Minimal kuota biasanya ±15 peserta (dapat disesuaikan). Mohon info estimasi peserta untuk proposal."},
    {"q": "Bagaimana alur permintaan in-house training?",
     "a": "Umumnya: kirim kebutuhan/topik & jumlah peserta → klarifikasi tujuan & profil peserta → proposal & jadwal → delivery."},
    {"q": "Apa faktor yang mempengaruhi harga in-house training?",
     "a": "Jumlah peserta, lokasi, materi/level kustomisasi, serta kebutuhan alat/praktik di lapangan."},
    {"q": "Bisakah materi disesuaikan dengan proses perusahaan?",
     "a": "Bisa. Kami dapat menyesuaikan studi kasus/contoh dengan proses dan isu di lini Anda."},
])

courses.to_csv("courses.csv", index=False)
faqs.to_csv("faqs.csv", index=False)

# ---------- 4) RAG index (sklearn, tanpa FAISS) ----------
courses = pd.read_csv("courses.csv")
faqs = pd.read_csv("faqs.csv")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_corpus(courses_df, faqs_df):
    docs, meta = [], []
    for _, r in courses_df.iterrows():
        t = (f"[COURSE] {r['code']} {r['title']} | Audience: {r['audience']} | "
             f"Format: {r['format']} | Durasi: {r['duration_days']} hari | "
             f"Jadwal: {r['next_runs']} | Lokasi: {r['location']} | "
             f"Harga: Rp{int(r['price_idr']):,} | Prasyarat: {r['prereq']} | "
             f"Deskripsi: {r['description']}")
        docs.append(t); meta.append({"type":"course","code":r["code"]})
    for i, r in faqs_df.iterrows():
        t = f"[FAQ] Q: {r['q']} A: {r['a']}"
        docs.append(t); meta.append({"type":"faq","id":i})
    return docs, meta

docs, meta = build_corpus(courses, faqs)
emb = embed_model.encode(docs, normalize_embeddings=True).astype("float32")
_nn = NearestNeighbors(metric="cosine").fit(emb)

def rag_search(query, k=5):
    qv = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    dists, ids = _nn.kneighbors(qv, n_neighbors=min(k, len(docs)), return_distance=True)
    out = []
    for d, i in zip(dists[0], ids[0]):
        sim = 1.0 - float(d)
        out.append((sim, docs[int(i)], meta[int(i)]))
    return out


def summarize_rag_results(rag_results, limit=3):
    """Return human-friendly bullet snippets for the top-k RAG hits."""
    snippets = []
    titles = []
    for score, _, meta_info in rag_results[:limit]:
        if meta_info["type"] == "course":
            r = courses[courses.code == meta_info["code"]].iloc[0]
            price = f"Rp{int(r.price_idr):,}".replace(",", ".")
            snippets.append(
                f"• {r.code} – {r.title}: {r.format}, {r.duration_days} hari. "
                f"Jadwal {r.next_runs} @ {r.location}. Harga {price}."
            )
            titles.append(f"{r.code} – {r.title}")
        else:
            faq_row = faqs.iloc[meta_info["id"]]
            snippets.append(f"• {faq_row.q} → {faq_row.a}")
            titles.append(faq_row.q)
    return snippets, titles

# ---------- 5) Intent classifier (TLC extended) + slot extractor ----------
train = [
    # Catalog
    ("Tunjukkan katalog pelatihan", "catalog"),
    ("List program untuk supplier", "catalog"),
    ("Training apa saja yang tersedia?", "catalog"),
    ("What trainings do you offer?", "catalog"),

    # Schedule
    ("Kapan jadwal JKK terdekat?", "schedule"),
    ("Next run TClass kapan?", "schedule"),
    ("When is the next session?", "schedule"),

    # Pricing
    ("Berapa biaya pelatihan JKK?", "pricing"),
    ("Harga TClass berapa?", "pricing"),
    ("Price per participant?", "pricing"),

    # Registration
    ("Saya mau daftar 10 peserta JKK", "registration"),
    ("Bagaimana cara mendaftar pelatihan?", "registration"),
    ("How do I enroll?", "registration"),

    # Custom / In-house
    ("Bisa training di lokasi supplier kami?", "custom"),
    ("Apakah bisa in-house training?", "custom"),
    ("Can you deliver on-site custom training?", "custom"),

    # External / In-house request
    ("Kami ingin mengajukan in-house training JKK", "external_training_request"),
    ("Apakah TLC bisa datang ke pabrik kami untuk training?", "external_training_request"),
    ("TLC bisa nggak adain pelatihan di kantor kami?", "external_training_request"),
    ("We need an on-site training for our company", "external_training_request"),
    ("Interested in in-house program for 25 pax", "external_training_request"),

    # Policy / payment
    ("Kebijakan pembatalan bagaimana?", "policy"),
    ("Metode pembayaran invoice apa?", "policy"),
    ("How to pay the invoice?", "policy"),

    # Contact
    ("Siapa yang bisa saya hubungi?", "contact"),
    ("Nomor WA TLC berapa?", "contact"),
    ("Who can I contact?", "contact"),

    # About TLC
    ("Apa itu Toyota Learning Center?", "about_tlc"),
    ("TLC itu bagian dari TMMIN ya?", "about_tlc"),
    ("Is TLC part of Corporate University?", "about_tlc"),

    # Program Focus
    ("Ada training untuk Safety?", "program_focus"),
    ("Pelatihan Leadership ada?", "program_focus"),
    ("Do you have Quality Core Tools?", "program_focus"),
    ("Digital learning program available?", "program_focus"),

    # Trainer Info
    ("Siapa pengajar JKK?", "trainer_info"),
    ("Apakah ada trainer dari TIA?", "trainer_info"),
    ("Who are the instructors?", "trainer_info"),

    # Certification
    ("Apakah dapat sertifikat?", "certification"),
    ("Is the certificate recognized?", "certification"),
    ("Sertifikat training diakui Corporate University?", "certification"),

    # Venue / Facility
    ("Apakah ada parkir di TLC Sunter?", "venue_facility"),
    ("Fasilitas training apa saja di TLC?", "venue_facility"),
    ("Is lunch provided?", "venue_facility"),

    # Vendor Support
    ("Bagaimana proses PO vendor?", "support_vendor"),
    ("Apakah bisa e-invoice?", "support_vendor"),
    ("Do you support vendor onboarding?", "support_vendor"),

    # FAQ General
    ("Training mulai jam berapa?", "faq_general"),
    ("Durasi training berapa lama?", "faq_general"),
    ("What time does the class start?", "faq_general"),
]
X = [t[0] for t in train]
y = [t[1] for t in train]
intent_clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
    ("logreg", LogisticRegression(max_iter=1000))
]).fit(X, y)

INTENT_PROFILES = [
    {"label": "catalog", "examples": ["katalog", "list program", "apa saja kursusnya", "training tersedia"], "description": "Permintaan daftar program/kursus"},
    {"label": "schedule", "examples": ["jadwal", "kapan kelas", "batch berikut", "tanggal training"], "description": "Menanyakan jadwal atau tanggal kelas"},
    {"label": "pricing", "examples": ["harga", "biaya", "fee", "tarif"], "description": "Menanyakan harga/biaya pelatihan"},
    {"label": "registration", "examples": ["daftar", "registrasi", "enroll", "ikut training"], "description": "Cara mendaftar atau permintaan pendaftaran"},
    {"label": "custom", "examples": ["in-house", "onsite", "ke pabrik", "private"], "description": "Meminta pelatihan khusus di lokasi peserta"},
    {"label": "external_training_request", "examples": ["in-house training", "on-site di pabrik", "datang ke kantor kami", "inhouse untuk company", "request external training"], "description": "Permintaan training eksternal/in-house untuk perusahaan"},
    {"label": "policy", "examples": ["batal", "refund", "pembayaran", "invoice"], "description": "Kebijakan pembatalan atau cara bayar"},
    {"label": "contact", "examples": ["hubungi", "nomor wa", "email", "kontak"], "description": "Permintaan kontak"},
    {"label": "about_tlc", "examples": ["apa itu tlc", "tmm in", "corporate university"], "description": "Menanyakan profil TLC"},
    {"label": "program_focus", "examples": ["ada training safety", "leadership ada", "quality tools"], "description": "Menanyakan fokus/tema program"},
    {"label": "trainer_info", "examples": ["siapa trainer", "pengajar", "instructor"], "description": "Menanyakan profil trainer"},
    {"label": "certification", "examples": ["sertifikat", "certificate", "diakui"], "description": "Pertanyaan sertifikasi"},
    {"label": "venue_facility", "examples": ["fasilitas", "parkir", "makan siang"], "description": "Fasilitas/venue"},
    {"label": "support_vendor", "examples": ["PO", "vendor onboarding", "e-invoice"], "description": "Dukungan administrasi vendor"},
    {"label": "faq_general", "examples": ["mulai jam berapa", "durasi", "berapa lama"], "description": "Pertanyaan umum jadwal/operasional"},
]
def build_intent_vectors():
    vectors = {}
    for profile in INTENT_PROFILES:
        seeds = profile["examples"] + [profile["description"]]
        emb_matrix = embed_model.encode(seeds, normalize_embeddings=True).astype("float32")
        avg_vec = np.mean(emb_matrix, axis=0)
        norm = np.linalg.norm(avg_vec)
        if norm:
            avg_vec = (avg_vec / norm).astype("float32")
        vectors[profile["label"]] = avg_vec
    return vectors

intent_vectors = build_intent_vectors()

def semantic_intent(text:str):
    qv = embed_model.encode([text], normalize_embeddings=True).astype("float32")[0]
    best_label, best_score = None, -1.0
    for label, vec in intent_vectors.items():
        score = float(np.dot(qv, vec))
        if score > best_score:
            best_label, best_score = label, score
    return best_label, best_score

def detect_intent(text:str):
    """Return (intent, debug_info) to expose routing internals for future tuning."""
    debug = {
        "clf_confidence": None,
        "semantic_best_intent": None,
        "semantic_similarity": None,
    }
    best_clf_intent, clf_conf = None, 0.0
    try:
        probs = intent_clf.predict_proba([text])[0]
        classes = intent_clf.named_steps["logreg"].classes_
        best_idx = int(np.argmax(probs))
        best_clf_intent = classes[best_idx]
        clf_conf = float(probs[best_idx])
        debug["clf_confidence"] = clf_conf
    except Exception:
        pass

    intent_sem, sem_score = semantic_intent(text)
    debug["semantic_best_intent"] = intent_sem
    debug["semantic_similarity"] = sem_score

    if best_clf_intent and clf_conf >= INTENT_CONFIDENCE_THRESHOLD:
        chosen = best_clf_intent
    elif intent_sem and sem_score >= SEMANTIC_INTENT_THRESHOLD:
        chosen = intent_sem
    else:
        chosen = None

    debug["final_intent"] = chosen
    return chosen, debug

# slots
DATE_RX = re.compile(r"(20\d{2}-\d{2}-\d{2})|(\d{1,2}\s*(Nov|Dec|Jan|Feb|Mar|Apr|Mei|May|Jun|Jul|Aug|Sep|Okt|Oct)\s*20\d{2})", re.I)
PAX_RX  = re.compile(r"(\d+)\s*(pax|orang|peserta|people)", re.I)
BARE_PAX_RX = re.compile(r"^\s*(\d{1,3})\s*$")
COURSE_RX = re.compile(r"(JKK[-\s]?\w+|TCLASS[-\s]?\w+|QCC[-\s]?\w+|XEV[-\s]?\w+|SAFETY[-\s]?\w+)", re.I)
COMPANY_RX = re.compile(r"(PT\s+[A-Za-z0-9.&()\-\s]+)", re.I)
LOCATION_RX = re.compile(
    r"(karawang|sunter|jakarta|bandung|bekasi|cikarang|purwakarta|cibitung|cikande|depok|bogor|tangerang|semarang|surabaya|bali|yogyakarta|jogja|medan|makassar|batam|balikpapan|samarinda)",
    re.I,
)

def extract_slots(text):
    slots = {}
    if not isinstance(text, str): 
        return slots
    m = PAX_RX.search(text);     slots["pax"]    = int(re.sub(r"\D","",m.group(1))) if m else None
    if not slots.get("pax") and BARE_PAX_RX.match(text.strip()):
        slots["pax"] = int(text.strip())
    m = COURSE_RX.search(text);  slots["course"] = m.group(0).upper().replace(" ","") if m else None
    m = DATE_RX.search(text);    slots["date"]   = m.group(0) if m else None
    m = COMPANY_RX.search(text); slots["company"]= m.group(0).strip() if m else None
    m = LOCATION_RX.search(text); slots["location"] = m.group(0).title().strip() if m else None
    return {k:v for k,v in slots.items() if v}

# ---------- 5b) Session state helpers ----------
SESSION_STATE_TEMPLATE = {
    "current_intent": None,
    "current_course_code": None,
    "current_course_title": None,
    "participants": None,
    "preferred_dates": None,
    "company_name": None,
    "location": None,
    "mode": None,  # "internal" | "external"
}

def init_session_state():
    return SESSION_STATE_TEMPLATE.copy()

def ensure_session_state(state):
    base = init_session_state()
    if isinstance(state, dict):
        for k in base:
            if k in state and state[k] is not None:
                base[k] = state[k]
    return base

def match_course_reference(text):
    if not text:
        return None
    text_lower = text.lower()
    slots = extract_slots(text)
    if slots.get("course"):
        code = slots["course"]
        match = courses[courses.code.str.contains(code.split("-")[0], case=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            return {"code": row.code, "title": row.title}
    # fallback: title keywords
    for _, row in courses.iterrows():
        if row.title.lower() in text_lower:
            return {"code": row.code, "title": row.title}
    return None

def append_unique_date(state, date_str):
    if not date_str:
        return
    if state.get("preferred_dates") is None:
        state["preferred_dates"] = []
    if date_str not in state["preferred_dates"]:
        state["preferred_dates"].append(date_str)

def apply_context_rules(text, intent, debug_info, state, slots):
    """Lightweight heuristic rules to make turn-level intent selection context aware."""
    normalized = text.lower()
    clf_conf = debug_info.get("clf_confidence") or 0.0
    low_conf = (intent is None) or (clf_conf < INTENT_CONFIDENCE_THRESHOLD)
    chosen = intent

    # Rule: if we recently discussed catalog/registration and pricing words appear, treat as pricing
    if low_conf and state.get("current_intent") in {"catalog", "registration"}:
        if any(word in normalized for word in PRICING_KEYWORDS):
            chosen = "pricing"

    # Rule: if user keeps talking about in-house/external and we already track that context, stay on it
    if low_conf and state.get("current_intent") == "external_training_request":
        if any(word in normalized for word in EXTERNAL_KEYWORDS):
            chosen = "external_training_request"

    # Rule: follow-up asking for in-house/external delivery keeps the course context
    if any(word in normalized for word in EXTERNAL_KEYWORDS):
        if state.get("current_course_code") or state.get("current_course_title") or slots.get("course"):
            chosen = "external_training_request"
            state["mode"] = "external"

    # Rule: if user shares lokasi + course hint with low confidence, assume external request
    if low_conf and (slots.get("location") or state.get("location")):
        if state.get("current_course_code") or state.get("current_course_title") or slots.get("course"):
            chosen = "external_training_request"
            state["mode"] = state.get("mode") or "external"

    # Rule: continuing registration/external flow with participant info should not drop the flow
    if state.get("current_intent") in {"registration", "custom", "external_training_request"} and slots.get("pax"):
        if chosen is None:
            chosen = state.get("current_intent")

    return chosen

def update_state_from_slots(state, slots, course_match=None):
    if not state:
        return
    if course_match:
        state["current_course_code"] = course_match.get("code")
        state["current_course_title"] = course_match.get("title")
    elif slots.get("course"):
        match = courses[courses.code.str.contains(slots["course"].split("-")[0], case=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            state["current_course_code"] = row.code
            state["current_course_title"] = row.title
    if slots.get("pax"):
        state["participants"] = slots["pax"]
    if slots.get("company"):
        state["company_name"] = slots["company"]
    if slots.get("location"):
        state["location"] = slots["location"]
    if slots.get("date"):
        append_unique_date(state, slots["date"])


def detect_missing_slots(slots, state):
    """Check which key details are still missing for guidance prompts."""
    missing = []
    if not (slots.get("course") or state.get("current_course_code") or state.get("current_course_title")):
        missing.append("course")
    if not (slots.get("company") or state.get("company_name")):
        missing.append("company_name")
    if not (slots.get("pax") or state.get("participants")):
        missing.append("participants")
    if not (slots.get("location") or state.get("location")):
        missing.append("location")
    if not (slots.get("date") or (state.get("preferred_dates") and len(state.get("preferred_dates")) > 0)):
        missing.append("preferred_dates")
    return missing

# ---------- 6) Handlers ----------
def render_course_row(r):
    price = f"Rp{int(r.price_idr):,}".replace(",",".")
    return f"- **{r.code} – {r.title}** ({r.format}, {r.duration_days} hari) | Jadwal: {r.next_runs} | Lokasi: {r.location} | Harga: {price}"

def handle_catalog():
    rows = "\n".join(render_course_row(r) for _, r in courses.iterrows())
    return f"Berikut program yang tersedia:\n{rows}\n\nButuh brosur PDF atau kurasi untuk supplier tertentu? Ketik: *rekomendasi untuk supplier komponen stamping*."

def handle_schedule(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    code_hint = slots.get("course")
    if not code_hint and session_state:
        code_hint = session_state.get("current_course_code")
    if code_hint:
        c = courses[courses.code.str.contains(code_hint.split("-")[0], case=False, regex=False)]
        if c.empty:
            c = courses[courses.title.str.contains("JKK|TClass|QCC", case=False, regex=True)]
    else:
        c = courses
    msg = ["Jadwal terdekat:"]
    for _, r in c.iterrows():
        msg.append(f"- {r.code}: {r.next_runs} @ {r.location}")
    return "\n".join(msg)


def handle_pricing(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    code_hint = slots.get("course")
    if not code_hint and session_state:
        code_hint = session_state.get("current_course_code")
    if code_hint:
        c = courses[courses.code.str.contains(code_hint.split("-")[0], case=False, regex=False)]
        if c.empty:
            return "Mohon sebutkan kode atau judul pelatihan yang dimaksud (mis. JKK-SV-101)."
        r = c.iloc[0]
        note = ""
        if session_state and session_state.get("participants"):
            note = f" untuk {session_state['participants']} peserta"
        return f"Harga {r.code} – {r.title}{note}: Rp{int(r.price_idr):,} per peserta."
    if session_state and session_state.get("current_course_title"):
        return (f"Untuk kursus {session_state['current_course_code']} – {session_state['current_course_title']}, "
                "mohon konfirmasi jumlah peserta agar kami hitung ulang.")
    return "Harga per peserta tertera di katalog. Sebutkan kode/judul pelatihan untuk detail harga."


def handle_registration(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    pax = slots.get("pax") or (session_state.get("participants") if session_state else None) or "(jumlah peserta?)"
    course = slots.get("course") or (session_state.get("current_course_code") if session_state else None) or "(kode kursus?)"
    comp = slots.get("company") or (session_state.get("company_name") if session_state else None) or "(nama perusahaan?)"
    course_title = session_state.get("current_course_title") if session_state else None
    course_line = f"• Kursus: {course}"
    if course_title and course_title not in course:
        course_line += f" – {course_title}"
    return textwrap.dedent(f"""
    Baik, saya siapkan pendaftaran:
    • Perusahaan: {comp}
    {course_line}
    • Jumlah Peserta: {pax}
    • Tautan Registrasi: {REG_FORM_URL}
    • Bantuan WA: {WHATSAPP_LINK}

    Jika ingin daftar via Google Sheet, kirimkan email ke {CONTACT_EMAIL} dengan subject: "REG {course} – {comp}" beserta data peserta (Nama, Email, No. HP).
    """).strip()


def handle_custom(text, session_state=None, slots=None):
    slots = slots or extract_slots(text)
    comp = slots.get("company") or (session_state.get("company_name") if session_state else None) or "(nama perusahaan?)"
    course_desc = ""
    if session_state and session_state.get("current_course_code"):
        course_desc = f" Fokus kursus: {session_state['current_course_code']} – {(session_state.get('current_course_title') or '')}"
    mode = session_state.get("mode") if session_state else None
    mode_line = "Mode: external/in-house" if not mode else f"Mode: {mode.title()}"
    return textwrap.dedent(f"""
    In-house / on-site bisa 👍
    • Skop: sesuaikan kebutuhan proses di lini {comp if comp!='(nama perusahaan?)' else 'supplier'}{course_desc}
    • {mode_line}
    • Lead time: 2–3 minggu dari konfirmasi
    • Minimal kuota: 15 peserta (nego)
    • Kirim kebutuhan (tujuan, profil peserta, tanggal target) ke {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}.
    """).strip()


def handle_external_training_request(text, session_state=None, slots=None):
    """Respond to in-house/external training requests with slot echo + next questions."""
    slots = slots or extract_slots(text)
    state = ensure_session_state(session_state)
    state["mode"] = state.get("mode") or "external"

    course = slots.get("course") or state.get("current_course_code") or state.get("current_course_title") or "(kode/judul kursus?)"
    course_title = state.get("current_course_title")
    course_line = f"• Kursus/topik: {course}"
    if course_title and course_title not in course:
        course_line += f" – {course_title}"

    company = slots.get("company") or state.get("company_name") or "(nama perusahaan?)"
    pax = slots.get("pax") or state.get("participants") or "(jumlah peserta?)"
    loc = slots.get("location") or state.get("location") or "(lokasi/kota?)"
    dates = slots.get("date") or ", ".join(state.get("preferred_dates") or []) or "(waktu target?)"

    missing = detect_missing_slots(slots, state)
    followups = []
    if "company_name" in missing:
        followups.append("Nama perusahaan?")
    if "participants" in missing:
        followups.append("Estimasi jumlah peserta?")
    if "location" in missing:
        followups.append("Lokasi training (pabrik/kantor/hotel) dan kota?")
    if "preferred_dates" in missing:
        followups.append("Tanggal target atau bulan rencana?")
    if "course" in missing:
        followups.append("Topik atau kode kursus yang diinginkan?")

    lines = [
        "Siap, kami catat permintaan in-house / on-site training:",
        f"• Perusahaan: {company}",
        course_line,
        f"• Perkiraan peserta: {pax}",
        f"• Lokasi/area: {loc}",
        f"• Waktu target: {dates}",
        "",
        "Materi bisa disesuaikan dengan proses dan studi kasus Anda.",
    ]
    if followups:
        lines.append("Mohon info tambahan agar kami siapkan proposal:")
        for q in followups:
            lines.append(f"- {q}")
    else:
        lines.append(f"Tim TLC akan hubungi lewat email/WA yang Anda berikan ({CONTACT_EMAIL} / {WHATSAPP_LINK}).")

    return "\n".join(lines)

def handle_policy():
    hits = faqs[faqs.q.str.contains("pembatalan|kebijakan", case=False, regex=True)]
    return hits.iloc[0].a if not hits.empty else "Pembatalan ≤ D-5: full refund; D-4–D-2: 50%; D-1/no-show: no refund."

def handle_contact():
    return f"Hubungi kami di {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}. Lokasi utama: {', '.join(LOCATIONS)}."

def handle_about_tlc():
    return ("Toyota Learning Center (TLC) adalah unit pengembangan kompetensi di TMMIN yang "
            "menyelenggarakan program pembelajaran berbasis Toyota Way & Corporate University. "
            "Program tertentu dibuka untuk supplier & publik.")

def handle_program_focus():
    return ("Fokus program:\n"
            "• Quality & Problem Solving: JKK, QCC, 7QC Tools, A3/TBP.\n"
            "• Safety & Compliance: Basic Safety, Near-Miss, Risk Assessment.\n"
            "• Leadership: Supervisor Essentials, Coaching, Communication.\n"
            "• Digital Learning: TClass Admin/Authoring, LMS reporting.\n"
            "Butuh kurasi sesuai profil supplier? Ketik: *rekomendasi untuk [jenis proses]*.")

def handle_trainer_info():
    return ("Pengajar adalah praktisi TLC/TMMIN & kolaborator (TIA). "
            "Untuk JKK/QCC, trainer berpengalaman penerapan di shop-floor & coaching pasca kelas.")

def handle_certification():
    return ("Peserta yang lulus kehadiran & evaluasi menerima e-certificate TLC. "
            "Untuk program tertentu, sertifikat mencantumkan jam belajar & level kompetensi.")

def handle_venue_facility():
    return (f"Lokasi: {', '.join(LOCATIONS)}.\n"
            "Fasilitas: ruang kelas ber-AC, proyektor, area ibadah, parkir, coffee/tea. "
            "Snack & lunch opsional. Online via Microsoft Teams.")

def handle_support_vendor():
    return ("Administrasi vendor:\n"
            "• Quotation & invoice setelah konfirmasi batch.\n"
            "• Pembayaran: transfer/invoice, e-invoice untuk vendor terdaftar.\n"
            f"Kontak: {CONTACT_EMAIL} / {WHATSAPP_LINK}")

def handle_faq_general():
    return ("Umum:\n"
            "• Jam pelatihan: 08.00–16.00 WIB (half-day 08.00–12.00 atau 13.00–17.00).\n"
            "• Durasi: 0.5–1 hari (contoh katalog).\n"
            "• Mode: Offline TLC/On-site, atau Online via MS Teams.\n"
            "• Minimal kuota in-house: ±15 peserta (nego).")

def generate_fallback_response(user_text, rag_results, detected_slots, missing_slots):
    """RAG-driven fallback that also guides users to share missing details."""
    snippets, titles = summarize_rag_results(rag_results, limit=3)
    topic_hint = titles[0] if titles else "topik pelatihan"

    followups = []
    if "course" in missing_slots:
        followups.append("Sebutkan kode atau judul pelatihan yang dimaksud?")
    if "company_name" in missing_slots:
        followups.append("Nama perusahaan Anda?")
    if "participants" in missing_slots:
        followups.append("Estimasi jumlah peserta?")
    if "location" in missing_slots:
        followups.append("Lokasi training yang diinginkan (pabrik/kantor/hotel dan kota)?")
    if "preferred_dates" in missing_slots:
        followups.append("Tanggal target atau bulan rencana training?")

    header = f"Sepertinya terkait {topic_hint}." if topic_hint else "Berikut beberapa pilihan terkait:"
    response_lines = [header]
    if snippets:
        response_lines.append("🔎 Info relevan:")
        response_lines.extend(snippets)

    if followups:
        response_lines.append("Supaya saya bisa jawab lebih tepat, mohon bantu info:")
        for q in followups[:3]:
            response_lines.append(f"• {q}")
    else:
        response_lines.append("Ada detail lain yang perlu saya lengkapi?")

    return "\n".join(response_lines)

# ---------- 7) Orchestrator ----------
def respond(user_text, session_state=None):
    # Normalisasi input
    if not isinstance(user_text, str):
        user_text = "" if user_text is None else str(user_text)

    state = ensure_session_state(session_state)
    intent, debug_info = detect_intent(user_text)
    slots = extract_slots(user_text)
    course_match = match_course_reference(user_text)

    intent = apply_context_rules(user_text, intent, debug_info, state, slots)
    update_state_from_slots(state, slots, course_match)

    clf_conf = debug_info.get("clf_confidence") or 0.0
    sem_conf = debug_info.get("semantic_similarity") or 0.0
    low_conf_router = (clf_conf < INTENT_CONFIDENCE_THRESHOLD) and (sem_conf < SEMANTIC_INTENT_THRESHOLD)

    if intent:
        state["current_intent"] = intent
        if intent == "external_training_request":
            state["mode"] = state.get("mode") or "external"

    fallback_needed = (intent is None) or (intent == "other") or low_conf_router

    if not fallback_needed and intent == "catalog":
        reply = handle_catalog()
    elif not fallback_needed and intent == "schedule":
        reply = handle_schedule(user_text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "pricing":
        reply = handle_pricing(user_text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "registration":
        reply = handle_registration(user_text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "external_training_request":
        reply = handle_external_training_request(user_text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "custom":
        reply = handle_custom(user_text, session_state=state, slots=slots)
    elif not fallback_needed and intent == "policy":
        reply = handle_policy()
    elif not fallback_needed and intent == "contact":
        reply = handle_contact()
    elif not fallback_needed and intent == "about_tlc":
        reply = handle_about_tlc()
    elif not fallback_needed and intent == "program_focus":
        reply = handle_program_focus()
    elif not fallback_needed and intent == "trainer_info":
        reply = handle_trainer_info()
    elif not fallback_needed and intent == "certification":
        reply = handle_certification()
    elif not fallback_needed and intent == "venue_facility":
        reply = handle_venue_facility()
    elif not fallback_needed and intent == "support_vendor":
        reply = handle_support_vendor()
    elif not fallback_needed and intent == "faq_general":
        reply = handle_faq_general()
    else:
        rag_results = rag_search(user_text, k=5)
        detected_slots = slots.copy()
        if course_match:
            detected_slots["course_match"] = course_match
        missing_slots = detect_missing_slots(slots, state)
        reply = generate_fallback_response(user_text, rag_results, detected_slots, missing_slots)

    return reply, state

# ---------- 8) Leads logging ----------
LEADS_CSV = "leads.csv"
if not os.path.exists(LEADS_CSV):
    pd.DataFrame(columns=["ts","name","company","email","phone","interest","notes"]).to_csv(LEADS_CSV, index=False)

def save_lead(name, company, email, phone, interest, notes):
    try:
        df = pd.read_csv(LEADS_CSV)
    except Exception:
        df = pd.DataFrame(columns=["ts","name","company","email","phone","interest","notes"])
    df.loc[len(df)] = [datetime.now().isoformat(), name, company, email, phone, interest, notes]
    df.to_csv(LEADS_CSV, index=False)

# ---------- 9) UI Gradio (robust) ----------
quick_msgs = [
    "Lihat katalog pelatihan",
    "Jadwal terdekat JKK",
    "Harga TClass",
    "Daftar 10 pax untuk JKK-SV-101",
    "Bisa in-house di pabrik kami PT XYZ?",
    "Minta proposal in-house 20 orang di Karawang",
    "Kebijakan pembatalan",
    "Kontak & lokasi"
]

def chat_fn(history, message, session_state, name, company, email, phone, interest):
    # Paksa ke string supaya tidak bikin error
    msg_text = "" if message is None else (message if isinstance(message, str) else str(message))
    name     = "" if not isinstance(name, str) else name
    company  = "" if not isinstance(company, str) else company
    email    = "" if not isinstance(email, str) else email
    phone    = "" if not isinstance(phone, str) else phone
    interest = "" if not isinstance(interest, str) else interest
    state    = ensure_session_state(session_state)

    try:
        reply, updated_state = respond(msg_text, session_state=state)
        if not isinstance(reply, str):
            reply = str(reply)
    except Exception as e:
        # Jangan biarkan exception menerus ke ASGI
        reply = f"Maaf, terjadi error di server: {e!s}"
        updated_state = state

    # Logging leads: hanya kalau msg_text string dan bisa di-lower()
    try:
        lower = msg_text.lower() if isinstance(msg_text, str) else ""
        if interest and any(k in lower for k in ["daftar", "register", "enroll", "minat", "pesan", "order"]):
            save_lead(name or "-", company or "-", email or "-", phone or "-", interest or "-", msg_text)
    except Exception:
        pass  # jangan biarkan logging merusak chat

    history = (history or []) + [(msg_text, reply)]
    return history, updated_state, ""

with gr.Blocks(theme="soft") as demo:
    gr.Markdown(f"### 🤖 {ORG_NAME} — Training Assistant (Beta)\nDukungan: ID/EN • Timezone: {TZ}\n\n"
                f"[Form Registrasi]({REG_FORM_URL}) • [WhatsApp]({WHATSAPP_LINK})")

    with gr.Row():
        with gr.Column(scale=3):
            name = gr.Textbox(label="Nama (optional)")
            company = gr.Textbox(label="Perusahaan (optional)")
            email = gr.Textbox(label="Email (optional)")
            phone = gr.Textbox(label="No HP/WA (optional)")
            interest = gr.Dropdown(["","JKK","TClass","QCC","Custom In-house","Other"], label="Minat (opsional)", value="")
            gr.Markdown("**Quick prompts:**")
            qp = gr.Radio(choices=quick_msgs, label="Pilih contoh pesan")
            qp_btn = gr.Button("Kirim Quick Prompt")

        with gr.Column(scale=7):
            chat = gr.Chatbot(height=420)  # default tuple mode
            msg = gr.Textbox(label="Ketik pertanyaan Anda…")
            send = gr.Button("Kirim", variant="primary")

    session_state = gr.State(init_session_state())

    qp_btn.click(lambda c: c or "Lihat katalog pelatihan", qp, msg)
    send.click(chat_fn, inputs=[chat, msg, session_state, name, company, email, phone, interest],
               outputs=[chat, session_state, msg])
    msg.submit(chat_fn, inputs=[chat, msg, session_state, name, company, email, phone, interest],
               outputs=[chat, session_state, msg])

demo.launch(debug=False, share=True)