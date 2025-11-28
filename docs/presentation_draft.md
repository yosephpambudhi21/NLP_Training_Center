# Toyota Learning Center Chatbot — Presentation Draft

## Slide 1 — Title
- **Toyota Learning Center Chatbot**
- AI assistant for course discovery, pricing, scheduling, registration, and FAQs
- Presenter name, date

## Slide 2 — Agenda
- Motivation & goals
- Solution overview
- How it works (architecture)
- Demo walkthrough
- Results & next steps
- Q&A

## Slide 3 — Motivation & Goals
- Reduce response time for common course and policy questions
- Standardize answers across teams and channels
- Capture leads when users are ready to register
- Support both online and offline (Colab/local) usage

## Slide 4 — Solution Overview
- Gradio/FastAPI chatbot tailored to Toyota Learning Center workflows
- Hybrid intent router (classifier + semantic) to keep answers on-topic
- Retrieval-augmented responses over courses and FAQs
- Lightweight lead capture to log registration interest

## Slide 5 — Architecture at a Glance
- **Interface**: Gradio UI served via FastAPI/uvicorn
- **Intent layer**: TF–IDF + Logistic Regression with semantic fallback thresholds
- **RAG**: `sentence-transformers` embeddings + cosine NearestNeighbors over course/FAQ corpus
- **Slots & context**: regex/fuzzy extractors for course, participants, dates, company, mode (internal/external)
- **Leads**: appends structured rows to `leads.csv`

## Slide 6 — Data Sources
- `courses.csv`: catalog seeds (title, code, summary)
- `faqs.csv`: common policy and logistics Q&A
- User conversation context: recent intent + extracted slots

## Slide 7 — Key Features
- Catalog lookup with top-K retrieval and concise summaries
- Pricing and schedule answers reuse detected course/date/participant details
- External training handling confirms on-site/in-house requests
- Registration path prompts for missing slots and stores leads
- Fallback uses RAG when confidence is low, then asks guided follow-ups

## Slide 8 — Offline & Online Modes
- **Offline/Colab**: skips heavy installs, falls back to TF–IDF embeddings
- **Online/local**: loads `all-MiniLM-L6-v2` for semantic search
- Share-link option for environments without localhost access

## Slide 9 — Demo Plan
- Start app (Colab: `share=True` if needed) and open UI
- Show a catalog query ("courses about leadership") and highlight retrieved courses
- Ask pricing with a course name and participant count; confirm slot reuse
- Request an external/on-site session; show company/location capture
- Trigger registration intent and submit contact fields; point to `leads.csv`
- Demonstrate low-confidence fallback and how it proposes clarifying questions

## Slide 10 — Results & Quality
- Fast responses for frequent intents (catalog, pricing, schedule)
- Context-aware follow-ups reduce back-and-forth for registration
- Lead log provides a handoff path to sales/ops
- Evaluation harness available (`python tests/intent_eval.py`) for regression checks

## Slide 11 — Roadmap
- Expand course/FAQ corpus and add multilingual intent examples
- Plug into live course inventory and pricing APIs
- Add analytics on intent distribution and unresolved questions
- Harden authentication and rate limits for external deployments

## Slide 12 — How to Run
- Local: `python app2.py` after installing `requirements.txt`
- Colab/notebook: import/run `app.py` with `TLC_SKIP_PIP=1` to skip installs; use `share=True` when localhost blocked

## Slide 13 — Q&A
- Invite questions; segue to discussion prompts below

## Suggested Questions to Ask the Audience
- Which intents or workflows are most critical for launch (catalog, registration, pricing, external delivery)?
- What coverage gaps do you see in the current course and FAQ seed data?
- Are there compliance or approval rules we must enforce before giving pricing or schedules?
- Should lead capture write to an external CRM instead of CSV? Which fields are mandatory?
- How should we prioritize languages or regional variants?
- What SLAs or guardrails are expected for response accuracy and latency?
- Any security constraints around hosting (on-prem, VPC, authentication)?
- What metrics would be most useful for monitoring success (intent accuracy, lead volume, deflection rate)?
