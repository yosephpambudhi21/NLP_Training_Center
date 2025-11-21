# Toyota Learning Center Chatbot — Technical Documentation

## Overview
This repository hosts two runnable chatbot entry points:
- **`app.py`** – notebook-friendly/Colab style launcher with optional dependency bootstrapping guard (`TLC_SKIP_PIP`).
- **`app2.py`** – local/runtime-oriented launcher that assumes dependencies are preinstalled.

Both variants share the same core logic for intent detection, slot extraction, RAG search, conversational state, and handlers tuned for Toyota Learning Center (TLC) training workflows.

## Architecture
### Data layer
- **Courses & FAQs**: Seeded inline `pandas` DataFrames provide example courses and FAQs that populate both the RAG corpus and handler responses (`courses`, `faqs`).
- **RAG corpus**: Built by combining course rows and FAQ entries, then encoded using `sentence-transformers/all-MiniLM-L6-v2` and indexed with `NearestNeighbors` (cosine metric) for top-k retrieval.

### Intent detection (hybrid)
- **Classifier**: TF-IDF + `LogisticRegression` pipeline trained on intent examples; exposes probabilities for confidence-based routing.
- **Semantic router**: Each intent has descriptive examples whose embeddings are averaged into an intent centroid. If classifier confidence is below `INTENT_CONFIDENCE_THRESHOLD`, semantic routing compares user text embedding against the centroids and applies `SEMANTIC_INTENT_THRESHOLD`.
- **API**: `detect_intent(text)` returns the chosen intent plus debug details: classifier confidence, best semantic match, and final decision.

### Slot extraction
Pattern-based extractors capture:
- Course codes/titles (with fuzzy matching via `match_course_reference`).
- Participant counts, preferred dates/months, company names, and external/in-house indicators.
These slots are used by handlers and to update session context.

### Session state & context rules
`gr.State` holds a per-conversation dictionary with the latest intent, course, participant counts, dates, company, and mode (internal/external). Context heuristics adjust routing when confidence is low, e.g. forcing `pricing` if pricing keywords appear after catalog/registration, or preserving external-training mode when users ask about on-site delivery.

### Handlers
- **Catalog / Schedule / Pricing / Registration**: Render tailored responses that reuse stored slots (course, participants, preferred dates) when available.
- **External training request**: Confirms in-house/on-site intent, echoes detected company/course/participants/location/timeframe, and asks for missing details.
- **Custom / Policy / Contact**: Provide targeted replies or FAQ snippets.
- **Fallback**: When intent is `other` or confidence is low, runs RAG, summarizes the top results, and poses guided follow-up questions for missing slots.

### Lead capture
If a message indicates registration intent and the user filled contact fields, `leads.csv` is appended with a timestamped entry. Lead logging is shared by both app variants.

### Evaluation harness
`tests/intent_eval.py` offers a CLI to benchmark intent and slot quality. It loads the same detection and extraction utilities, runs hard-coded test cases, reports overall/per-intent accuracy, and can list misclassified examples. Run via `python tests/intent_eval.py`.

## Configuration
Key tuning knobs in both apps:
- `INTENT_CONFIDENCE_THRESHOLD` and `SEMANTIC_INTENT_THRESHOLD` control classifier vs. semantic routing strictness.
- `PRICING_KEYWORDS` / external-training keywords drive context overrides.
- Contact metadata (`ORG_NAME`, `CONTACT_EMAIL`, `WHATSAPP_LINK`) and quick replies live near the top of each app file.

## Running the apps
1. Install dependencies from `requirements.txt` into a virtual environment.
2. For local use, run `python app2.py` and open the provided Gradio URL.
3. For notebook/Colab usage, set `TLC_SKIP_PIP=1` when importing to avoid bootstrapping; otherwise run `python app.py`.

## Extensibility notes
- Add intents by expanding training pairs, semantic profiles, and handler mappings in both app files.
- Extend slot patterns or context rules where noted in the code comments above each heuristic block.
- Grow the evaluation set in `tests/intent_eval.py` to track regression or improvement across intents/slots.
