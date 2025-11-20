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
    {"label":"catalog", "examples":["katalog","list program","apa saja kursusnya","training tersedia"], "description":"Permintaan daftar program/kursus"},
    {"label":"schedule","examples":["jadwal","kapan kelas","batch berikut","tanggal training"], "description":"Menanyakan jadwal atau tanggal kelas"},
    {"label":"pricing","examples":["harga","biaya","fee","tarif"], "description":"Menanyakan harga/biaya pelatihan"},
    {"label":"registration","examples":["daftar","registrasi","enroll","ikut training"], "description":"Cara mendaftar atau permintaan pendaftaran"},
    {"label":"custom","examples":["in-house","onsite","ke pabrik","private"], "description":"Meminta pelatihan khusus di lokasi peserta"},
    {"label":"policy","examples":["batal","refund","pembayaran","invoice"], "description":"Kebijakan pembatalan atau cara bayar"},
    {"label":"contact","examples":["hubungi","nomor wa","email","kontak"], "description":"Permintaan kontak"},
    {"label":"about_tlc","examples":["apa itu tlc","tmm in","corporate university"], "description":"Menanyakan profil TLC"},
    {"label":"program_focus","examples":["ada training safety","leadership ada","quality tools"], "description":"Menanyakan fokus/tema program"},
    {"label":"trainer_info","examples":["siapa trainer","pengajar","instructor"], "description":"Menanyakan profil trainer"},
    {"label":"certification","examples":["sertifikat","certificate","diakui"], "description":"Pertanyaan sertifikasi"},
    {"label":"venue_facility","examples":["fasilitas","parkir","makan siang"], "description":"Fasilitas/venue"},
    {"label":"support_vendor","examples":["PO","vendor onboarding","e-invoice"], "description":"Dukungan administrasi vendor"},
    {"label":"faq_general","examples":["mulai jam berapa","durasi","berapa lama"], "description":"Pertanyaan umum jadwal/operasional"},
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
COURSE_RX = re.compile(r"(JKK[-\s]?\w+|TCLASS[-\s]?\w+|QCC[-\s]?\w+|XEV[-\s]?\w+|SAFETY[-\s]?\w+)", re.I)
COMPANY_RX = re.compile(r"(PT\s+[A-Za-z0-9.&()\-\s]+)", re.I)

def extract_slots(text):
    slots = {}
    if not isinstance(text, str): 
        return slots
    m = PAX_RX.search(text);     slots["pax"]    = int(re.sub(r"\D","",m.group(1))) if m else None
    m = COURSE_RX.search(text);  slots["course"] = m.group(0).upper().replace(" ","") if m else None
    m = DATE_RX.search(text);    slots["date"]   = m.group(0) if m else None
    m = COMPANY_RX.search(text); slots["company"]= m.group(0).strip() if m else None
    return {k:v for k,v in slots.items() if v}

# ---------- 6) Handlers ----------
def render_course_row(r):
    price = f"Rp{int(r.price_idr):,}".replace(",",".")
    return f"- **{r.code} – {r.title}** ({r.format}, {r.duration_days} hari) | Jadwal: {r.next_runs} | Lokasi: {r.location} | Harga: {price}"

def handle_catalog():
    rows = "\n".join(render_course_row(r) for _, r in courses.iterrows())
    return f"Berikut program yang tersedia:\n{rows}\n\nButuh brosur PDF atau kurasi untuk supplier tertentu? Ketik: *rekomendasi untuk supplier komponen stamping*."

def handle_schedule(text):
    slots = extract_slots(text)
    if slots.get("course"):
        c = courses[courses.code.str.contains(slots["course"].split("-")[0], case=False, regex=False)]
        if c.empty:
            c = courses[courses.title.str.contains("JKK|TClass|QCC", case=False, regex=True)]
    else:
        c = courses
    msg = ["Jadwal terdekat:"]
    for _, r in c.iterrows():
        msg.append(f"- {r.code}: {r.next_runs} @ {r.location}")
    return "\n".join(msg)

def handle_pricing(text):
    slots = extract_slots(text)
    if slots.get("course"):
        c = courses[courses.code.str.contains(slots["course"].split("-")[0], case=False)]
        if c.empty: 
            return "Mohon sebutkan kode atau judul pelatihan yang dimaksud (mis. JKK-SV-101)."
        r = c.iloc[0]
        return f"Harga {r.code} – {r.title}: Rp{int(r.price_idr):,} per peserta."
    return "Harga per peserta tertera di katalog. Sebutkan kode/judul pelatihan untuk detail harga."

def handle_registration(text):
    slots = extract_slots(text)
    pax = slots.get("pax","(jumlah peserta?)")
    course = slots.get("course","(kode kursus?)")
    comp = slots.get("company","(nama perusahaan?)")
    return textwrap.dedent(f"""
    Baik, saya siapkan pendaftaran:
    • Perusahaan: {comp}
    • Kursus: {course}
    • Jumlah Peserta: {pax}
    • Tautan Registrasi: {REG_FORM_URL}
    • Bantuan WA: {WHATSAPP_LINK}
    
    Jika ingin daftar via Google Sheet, kirimkan email ke {CONTACT_EMAIL} dengan subject: "REG {course} – {comp}" beserta data peserta (Nama, Email, No. HP).
    """).strip()

def handle_custom(text):
    slots = extract_slots(text)
    comp = slots.get("company","(nama perusahaan?)")
    return textwrap.dedent(f"""
    In-house / on-site bisa 👍
    • Skop: sesuaikan kebutuhan proses di lini {comp if comp!='(nama perusahaan?)' else 'supplier'} (audit standard work, PDCA, QCC, TClass admin, dsb.)
    • Lead time: 2–3 minggu dari konfirmasi
    • Minimal kuota: 15 peserta (nego)
    • Kirim kebutuhan (tujuan, profil peserta, tanggal target) ke {CONTACT_EMAIL} atau WA {WHATSAPP_LINK}.
    """).strip()

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

def handle_rag(text):
    results = rag_search(text, k=5)
    tops = ["🔎 Saya temukan info relevan:"]
    for sc, _, meta_info in results[:4]:
        if meta_info["type"] == "course":
            r = courses[courses.code == meta_info["code"]].iloc[0]
            price = f"Rp{int(r.price_idr):,}".replace(",", ".")
            tops.append(
                f"• {r.code} – {r.title}: {r.format}, {r.duration_days} hari. "
                f"Jadwal {r.next_runs} @ {r.location}. Harga {price}."
            )
        else:
            faq_row = faqs.iloc[meta_info["id"]]
            tops.append(f"• {faq_row.q} → {faq_row.a}")
    tops.append("Kalimatmu belum spesifik? Sertakan kode kursus/tema atau jumlah peserta agar jawaban lebih tepat.")
    return "\n".join(tops)

# ---------- 7) Orchestrator ----------
def respond(user_text):
    # Normalisasi input
    if not isinstance(user_text, str):
        user_text = "" if user_text is None else str(user_text)

    intent, _debug = detect_intent(user_text)

    if intent == "catalog":        return handle_catalog()
    if intent == "schedule":       return handle_schedule(user_text)
    if intent == "pricing":        return handle_pricing(user_text)
    if intent == "registration":   return handle_registration(user_text)
    if intent == "custom":         return handle_custom(user_text)
    if intent == "policy":         return handle_policy()
    if intent == "contact":        return handle_contact()
    if intent == "about_tlc":      return handle_about_tlc()
    if intent == "program_focus":  return handle_program_focus()
    if intent == "trainer_info":   return handle_trainer_info()
    if intent == "certification":  return handle_certification()
    if intent == "venue_facility": return handle_venue_facility()
    if intent == "support_vendor": return handle_support_vendor()
    if intent == "faq_general":    return handle_faq_general()

    # fallback
    out = handle_rag(user_text)
    return out if isinstance(out, str) else str(out)

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
    "Kebijakan pembatalan",
    "Kontak & lokasi"
]

def chat_fn(history, message, name, company, email, phone, interest):
    # Paksa ke string supaya tidak bikin error
    msg_text = "" if message is None else (message if isinstance(message, str) else str(message))
    name     = "" if not isinstance(name, str) else name
    company  = "" if not isinstance(company, str) else company
    email    = "" if not isinstance(email, str) else email
    phone    = "" if not isinstance(phone, str) else phone
    interest = "" if not isinstance(interest, str) else interest

    try:
        reply = respond(msg_text)
        if not isinstance(reply, str):
            reply = str(reply)
    except Exception as e:
        # Jangan biarkan exception menerus ke ASGI
        reply = f"Maaf, terjadi error di server: {e!s}"

    # Logging leads: hanya kalau msg_text string dan bisa di-lower()
    try:
        lower = msg_text.lower() if isinstance(msg_text, str) else ""
        if interest and any(k in lower for k in ["daftar", "register", "enroll", "minat", "pesan", "order"]):
            save_lead(name or "-", company or "-", email or "-", phone or "-", interest or "-", msg_text)
    except Exception:
        pass  # jangan biarkan logging merusak chat

    history = (history or []) + [(msg_text, reply)]
    return history, ""

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

    state = gr.State([])

    qp_btn.click(lambda c: c or "Lihat katalog pelatihan", qp, msg)
    send.click(chat_fn, inputs=[state, msg, name, company, email, phone, interest],
               outputs=[chat, msg])
    msg.submit(chat_fn, inputs=[state, msg, name, company, email, phone, interest],
               outputs=[chat, msg])

demo.launch(debug=False, share=True)