---

# rag-openagenda

RAG proof-of-concept that recommends/answers questions about **public cultural events** using the OpenDataSoft **OpenAgenda** dataset. The system ingests city-filtered events from the last **12 months** (and any future events already listed), builds a **FAISS** vector index with **Mistral** embeddings, and answers via a retrieval-first chatbot orchestrated with **LangChain**.

* **Scope:** city-level (choose the city at run time)
* **Freshness rule:** keep events with `end_utc` (preferred) or `start_utc` ≥ **now − 365 days**
* **No hallucinations:** answers come only from retrieved context; URLs are never fabricated
* **Dates retrieval boost:** normalized date tokens (e.g., `YearStart: 2025`) embedded into text
* **“Upcoming”** = from **tomorrow (Europe/Paris)** onward when requested
* **Language:** auto-detects question language; answers in that language (fallback configurable)

---

## Dataset

* **OpenDataSoft dataset name:** `evenements-publics-openagenda`
* **API base:** `https://public.opendatasoft.com/api/records/1.0/search/`
* We use a **city filter** only and respect the platform’s **10k results window** (offset + rows ≤ 10 000).

---

## Architecture (high level)

1. **Fetcher** (`src/data/fetch_openagenda.py`)

   * City filter, safe pagination, last-12-months filter, HTML → text, URL normalization
   * Output schema: `uid, title, text, city, venue, start_utc, end_utc, tags, website, permalink`

2. **Indexer** (`src/index/build_index.py`)

   * Builds LangChain `Document`s and embeds **normalized date tokens** into `page_content`
   * Mistral embeddings → FAISS (saved locally)

3. **Chat** (`src/chat_demo.py`)

   * Retrieval-first, month-aware query augmentation, optional “upcoming” filter (Paris time), link sanitizer

4. **Pipeline** (`src/run_pipeline.py`)

   * One-shot: fetch → index → chat with the same behavior as the demo

---

## Requirements

* **Python 3.10+**
* A **Mistral API key** (for embeddings & chat)
* OS tested on macOS; Linux/Windows should work similarly

---

## Setup

```bash
# from project root
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (root of the repo):

```bash
cp .env.example .env
# then edit .env and set:
# MISTRAL_API_KEY=your_actual_key_here
```

> If you see a tokenizer warning from `langchain_mistralai`, it’s harmless.

---
# workflow

flowchart TD
  A([CLI]):::cli --> A1{Mode?}
  A1 -->|One-shot| A2[run_pipeline.py]
  A1 -->|Build + Chat| A3[build_index.py] --> A4[chat_demo.py]

  %% ========== INGEST ==========
  subgraph INGEST
    direction LR
    B[fetch_openagenda.py\n(city-only, safe pagination)] --> C[OpenDataSoft API\n"evenements-publics-openagenda"]
    C --> D{≤ 12 months?\nend_utc ≥ now-365d OR start_utc ≥ now-365d}
    D -->|yes| E[Clean HTML, normalize URLs,\nselect schema: title/text/city/venue/dates/tags/link]
    D -->|no| X[Drop record]
    E --> F[(DataFrame rows)]
  end

  %% ========== INDEX ==========
  subgraph INDEX
    direction LR
    F --> G[Build LangChain Documents\n+ embed normalized date tokens:\nYearStart/MonthStart/YearEnd/MonthEnd]
    G --> H[Split (RecursiveCharacterTextSplitter)]
    H --> I[Embeddings: Mistral "mistral-embed"]
    I --> J[FAISS index]
    J --> K{Persist?}
    K -->|--index-out| K1[Save to disk\n(data/index/faiss_*)]
    K -->|--ephemeral| K2[Keep in memory only]
  end

  %% ========== CHAT ==========
  subgraph CHAT
    direction TB
    L[chat_demo.py or run_pipeline.py loop] --> L1[Detect language (safe)\n+ fallback]
    L --> M{Month/Year mentioned?}
    M -->|yes| M1[Augment query with date tokens\n(YearStart/MonthStart/YearEnd/MonthEnd)]
    M -->|no| M2[Use raw query]
    M1 --> N[Retriever: similarity_search (k)]
    M2 --> N
    N --> O{Ask for "upcoming"/"à venir"?}
    O -->|yes| O1[Filter docs by\nParis tomorrow 00:00 (UTC)]
    O -->|no| O2[Use docs as-is]
    O1 --> P[Build CONTEXT + collect allowed URLs from metadata]
    O2 --> P
    P --> Q[LLM: Mistral chat\n(system forbids made-up links)]
    Q --> R[Sanitize links:\nstrip any URL not in allowed set]
    R --> S([Answer to user])
  end

  %% ========== TESTS ==========
  subgraph TESTS (required)
    direction LR
    K1 --> T[tests/test_index_data_constraints.py\nLoad index.pkl → ensure all docs are:\n• ≤ 12 months old\n• same (selected) city]
  end

  %% ========== CLEANUP (optional flags) ==========
  subgraph CLEANUP (optional)
    direction LR
    K1 --> U{--cleanup-index-out?}
    U -->|yes (and marker exists)| U1[Delete saved index dir]
    U -->|no| U2[Keep index on disk]
  end

  classDef cli fill:#222,color:#fff,stroke:#555,stroke-width:1;

---

## Build an index (per city)

```bash
# Paris
python -m src.index.build_index --city "Paris" \
  --out data/index/faiss_paris --max-records 9000

# Strasbourg
python -m src.index.build_index --city "Strasbourg" \
  --out data/index/faiss_strasbourg --max-records 9000
```

Notes:

* Use quotes for cities with spaces/hyphens: `"Aix-en-Provence"`, `"Saint-Étienne"`.
* `--max-records` keeps runtime reasonable while staying under the 10k window.

---

## Chat on a saved index

```bash
python -m src.chat_demo --index data/index/faiss_paris --k 20
```

Tips:

* **k**: increase to 20–40 for big cities or month-specific queries.
* Ask in any supported language (en/fr/de/es/ar/it/nl/pt).
* Use “**upcoming** / **à venir**” to restrict answers to **from tomorrow** onward (Paris time).

---

## One-shot pipeline (build + chat)

```bash
python -m src.run_pipeline --city "Paris" \
  --max-records 9000 \
  --index-out data/index/faiss_paris \
  --k 20
```

---

## “Upcoming”, language & links — behavior

* **Upcoming filter:** triggers when query contains *upcoming / coming up / à venir / prochains / bientôt / futurs*. It filters results to events whose **start or end** is **≥ tomorrow 00:00** **Europe/Paris**.
* **Language:** auto-detects the question’s language and answers in it (fallback: English; change with `--fallback-lang`).
* **Links:** only shows URLs that exist in the retrieved **context metadata** (`permalink`/`website`). Any model-invented links are stripped.

---

## Tests (project requirement)

This test verifies that **data integrated into the FAISS index**:

* is **≤ 12 months** old, and
* belongs to a **single selected city**.

1. Build the index first:

```bash
python -m src.index.build_index --city "Paris" --out data/index/faiss --max-records 9000
```

2. Run the test:

```bash
python -m pytest -q tests/test_index_data_constraints.py
```

If your index is in another folder:

```bash
INDEX_DIR="data/index/faiss_paris" python -m pytest -q tests/test_index_data_constraints.py
```

---

## Project structure

```
final-rag-openagenda/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .env                 # (not committed) contains MISTRAL_API_KEY=...
├─ data/
│  └─ index/
│     └─ faiss_*        # saved FAISS indexes (gitignored)
├─ src/
│  ├─ data/
│  │  └─ fetch_openagenda.py
│  ├─ index/
│  │  └─ build_index.py
│  ├─ chat_demo.py
│  └─ run_pipeline.py
├─ tests/
│  └─ test_index_data_constraints.py
└─ pytest.ini
```

---

## Troubleshooting

* **HTTP 400: “10000”**
  You hit the OpenDataSoft **10k window**. Our fetcher guards against this, but if you override params or request too many rows, reduce `--max-records` or keep the default pagination.

* **`ModuleNotFoundError: langchain_community` when testing**
  Always run pytest with the venv’s Python:

  ```bash
  python -m pytest -q tests/test_index_data_constraints.py
  ```

  Ensure `pip install -r requirements.txt` was done **inside** the venv.

* **Tokenizer warning from langchain\_mistralai**
  Safe to ignore. It falls back to a simple length heuristic for batching.

* **“I don’t know.” answers**
  Means the top-k retrieved context didn’t contain that fact within the last 12 months for the chosen city. Try increasing `--k` (e.g., 30–40) or adjusting the query (add a month/year).

---

## Deliverables mapping

* **Readme + dependency management** → `README.md`, `requirements.txt`, `.env(.example)`
* **Pre-processing** → `src/data/fetch_openagenda.py` (HTML clean, last-year filter, URL normalize)
* **Vectorization & index mgmt** → `src/index/build_index.py` (Mistral embeddings + FAISS)
* **RAG code** → `src/chat_demo.py`, `src/run_pipeline.py` (LangChain orchestration)
* **Unit test** (data constraints in vector DB) → `tests/test_index_data_constraints.py`

---

## License / usage

This POC is intended for educational/demo purposes. Respect OpenDataSoft/OpenAgenda terms when querying and redistributing data.

---

