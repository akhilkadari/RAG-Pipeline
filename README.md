# Helix RAG — Hybrid Retrieval over Internal Docs

A production-grade Retrieval-Augmented Generation system that ingests internal documentation, indexes it with **both dense vector and sparse keyword** search, retrieves the most relevant context for any question, and generates **grounded answers with inline citations** that are programmatically verified.

> Pitch line for interviews:
> *"I built a RAG system with hybrid search, citation verification, and a confidence scorer that tells you when to trust the answer. On a 50+ question eval suite, hybrid retrieval beat dense-only by **X%** on faithfulness and **Y%** on retrieval relevance."*
>
> (Run `python -m scripts.eval --mode hybrid` and `--mode dense` to fill in your own X / Y.)

---

## Why this project is interview-grade

Most RAG demos fall apart on three things. This project addresses each head-on:

| Production concern | What's typically skipped | What this project does |
|---|---|---|
| **Retrieval over technical docs** | Pure dense retrieval misses exact tokens (`SCHEMA_MISMATCH`, `helix-jit`, `429`) | Hybrid dense + BM25 with **Reciprocal Rank Fusion**, configurable weights, and an LLM-as-judge reranker |
| **Citation accuracy** | Models hallucinate citations — `[1]` doesn't actually support the claim | Every (claim, citation) pair is sent to a verifier judge; unsupported citations are flagged in the API response |
| **"I don't know"** | The model invents an answer when context is thin | Below-threshold retrieval triggers a structured fallback that names which documents *might* have the answer, instead of fabricating |
| **Chunking strategy choice** | "We just used the default" | Three switchable strategies (fixed-size, recursive, semantic) compared on the same eval suite |
| **Eval discipline** | One PDF, vibes-based testing | 50+ hand-written golden Q&As across lookup / multi-hop / no-answer / ambiguous categories with automated metrics |

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │          Streamlit Console          │
                    │  (chunks, citations, confidence,    │
                    │   hybrid vs dense side-by-side)     │
                    └─────────────────┬───────────────────┘
                                      │ HTTP
                    ┌─────────────────▼───────────────────┐
                    │         FastAPI service             │
                    │  /v1/ask  /v1/documents  /v1/ingest │
                    └─────────────────┬───────────────────┘
                                      │
            ┌─────────────────────────┴────────────────────────┐
            │                  RAG Pipeline                    │
            │  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
            │  │  Retrieval   │→ │ Generation │→ │ Verifier │  │
            │  │ dense+sparse │  │  GPT-4o    │  │  judge   │  │
            │  │  →RRF→rerank │  │  + prompt  │  │ +scorer  │  │
            │  └──────────────┘  └────────────┘  └──────────┘  │
            └────┬─────────────────┬──────────────────┬────────┘
                 │                 │                  │
        ┌────────▼─────┐  ┌────────▼─────┐  ┌────────▼─────┐
        │   ChromaDB   │  │ rank_bm25    │  │  OpenAI API  │
        │ (dense vec.) │  │ (sparse idx) │  │  (LLM+embed) │
        └──────────────┘  └──────────────┘  └──────────────┘
                 ▲                 ▲
                 │                 │
        ┌────────┴─────────────────┴────────┐
        │       Indexing Pipeline           │
        │  Documents → Chunks → Dedup       │
        │  → Embeddings → both indexes      │
        └────────┬──────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  Loaders (PDF,   │
        │  MD, HTML, TXT)  │
        │  raw → JSON      │
        └──────────────────┘
```

---

## Project layout

```
RAG Pipeline/
├── src/
│   ├── config.py                 # central settings (env vars / .env)
│   ├── loaders/                  # multi-format → uniform Document
│   ├── chunking/                 # 3 switchable strategies
│   ├── embeddings/               # OpenAI embedder w/ retry+batch
│   ├── indexing/                 # Chroma vector store + BM25 + dedup + pipeline
│   ├── retrieval/                # dense, sparse, RRF fusion, LLM reranker, hybrid orchestrator
│   ├── generation/               # grounded generator, citation verifier, confidence scorer
│   ├── pipeline.py               # high-level RAGPipeline (retrieval → gen → verify → score → IDK)
│   ├── api/                      # FastAPI app + Pydantic schemas
│   └── eval/                     # case loader, metrics, runner
├── scripts/
│   ├── ingest.py                 # raw → processed JSON
│   ├── build_index.py            # processed → chunks → embeddings → indexes
│   ├── seed.py                   # one-shot bootstrap from sample_corpus
│   ├── eval.py                   # run the golden Q&A suite
│   └── compare_chunking.py       # compare all 3 chunking strategies side-by-side
├── frontend/
│   └── app.py                    # Streamlit dashboard
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.frontend
├── docker-compose.yml
├── data/
│   ├── raw/sample_corpus/        # 6 fictional Helix Platform docs
│   ├── processed/                # normalized JSON Documents (re-indexable)
│   ├── eval/golden.json          # 50+ Q&A golden set
│   └── index/                    # ChromaDB + BM25 pickle (created at runtime)
├── tests/                        # unit tests (no API key required)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

### 1. Local Python setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

### 3. Seed the sample corpus and build the index

```bash
python -m scripts.seed
```

This ingests the 6 fictional docs in `data/raw/sample_corpus/`, normalizes them, chunks, embeds, dedups, and writes both Chroma and BM25 indexes under `data/index/`.

### 4. Ask a question end-to-end

```bash
python -c "
from src.pipeline import RAGPipeline
import json
r = RAGPipeline().ask('What is the Helix API rate limit?')
print(json.dumps(r.to_dict(), indent=2)[:2000])
"
```

### 5. Run the API + dashboard

```bash
# Terminal 1
uvicorn src.api.main:app --reload --port 8000
# Open http://localhost:8000/docs for OpenAPI

# Terminal 2
streamlit run frontend/app.py
# Open http://localhost:8501
```

### 6. Run the eval suite

```bash
python -m scripts.eval --mode hybrid
python -m scripts.eval --mode dense        # ablation
python -m scripts.compare_chunking         # all 3 chunkers head-to-head
```

The first eval run uses many LLM calls (judge per case) and will take several minutes. Results are written to `data/eval/last_run.json` and `data/eval/chunking_comparison.{json,md}`.

### 7. Or just `docker compose up`

```bash
cp .env.example .env  # ensure OPENAI_API_KEY is set
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

The compose stack auto-runs `scripts.seed` before launching the API.

---

## Phase-by-phase guide to the codebase

### Phase 1 — Ingestion & chunking

- `src/loaders/` — `BaseLoader` plus per-format implementations (`pypdf`, `markdown-it`, `BeautifulSoup`). Every loader returns a list of `Document` (`text` + `metadata`). Strict UTF-8 by default — bad encodings raise loudly so you find them in development.
- `src/chunking/` — three strategies:
  - `FixedSizeChunker` — char-window baseline (LangChain `CharacterTextSplitter`).
  - `RecursiveChunker` — heading-aware (`# ` → `## ` → paragraph → sentence).
  - `SemanticChunker` — embeds sentence neighbourhoods, splits at the 90th percentile of consecutive cosine distances.
- `src/indexing/pipeline.py` — `Documents → chunks → embed → dedup (cosine ≥ 0.95) → Chroma + BM25 in lockstep`. BM25 is rebuilt from the union of indexed chunks every run so the two indexes never drift.
- `src/indexing/deduplicator.py` — both exact hash and cosine near-duplicate detection.

### Phase 2 — Hybrid retrieval

- `DenseRetriever` queries Chroma; `SparseRetriever` queries BM25.
- `RRFFuser` merges them with $\text{score}(d) = \frac{w_\text{dense}}{k + r_\text{dense}(d)} + \frac{w_\text{sparse}}{k + r_\text{sparse}(d)}$, weights and `k` configurable via `.env`.
- `LLMReranker` re-scores the top 20 fused candidates 0–10 via a strict-format prompt, returns the top 5.
- `HybridRetriever.retrieve(..., mode=...)` exposes `hybrid` / `dense` / `sparse` for ablation studies.

### Phase 3 — Generation & verification

- `GroundedGenerator` formats numbered context blocks and uses a strict system prompt that mandates `[1]` / `[2, 3]` style citations and a fixed phrase for "no info".
- `CitationVerifier` splits the answer into sentences, extracts citations from each, and asks the judge model whether each (claim, cited passage) pair is *actually* supported. Returns coverage and accuracy.
- `ConfidenceScorer` weights retrieval, citation coverage, citation accuracy, and completeness (LLM-as-judge) into a composite score.
- `RAGPipeline` short-circuits to a structured "I don't know" response below `CONFIDENCE_THRESHOLD` and surfaces *which documents might be worth checking manually* — strictly more useful than a fabricated answer.

### Phase 4 — Evaluation framework

- `data/eval/golden.json` — 52 hand-written cases across:
  - **lookup** (single-fact questions)
  - **multi_hop** (require combining ≥ 2 docs)
  - **no_answer** (the corpus has no answer; the system must decline)
  - **ambiguous** (vague phrasing — does the system find the right interpretation?)
- `src/eval/metrics.py` — for each case, computes:
  - `correctness`: LLM-as-judge against the golden answer (special-cases no-answer)
  - `faithfulness`: are all claims grounded in retrieved context?
  - `retrieval_relevance`: did the expected source files appear in the retrieved chunks?
  - `citation_accuracy` / `citation_coverage`: from the verifier
  - `confidence_composite`: from the scorer
  - `idk_match_rate`: did the system correctly decline when it should have?
- `scripts/compare_chunking.py` — re-indexes under each chunker, runs the eval suite, and writes a comparison report (`data/eval/chunking_comparison.md`).

### Phase 5 — API + dashboard + Docker

- `src/api/main.py` — FastAPI with `POST /v1/ask`, `GET /v1/documents`, `POST /v1/ingest` (multipart), `GET /v1/health`. Auto-generated OpenAPI at `/docs`.
- `frontend/app.py` — Streamlit console with retrieved chunks, citation pass/fail per claim, confidence breakdown, and a **side-by-side hybrid vs dense-only toggle** for live demos.
- `docker-compose.yml` — `api` (auto-seeds at boot) + `frontend`.

### Phase 6 — Polish (manual)

- Record a < 4-minute demo: ingest, ask easy / multi-hop / no-answer questions, point at a hallucinated citation that the verifier catches, toggle dense vs hybrid.
- Add the eval numbers to your résumé bullet:
  > *"Achieved X% faithfulness and Y% citation accuracy on a 52-question eval; hybrid retrieval improved retrieval-relevance by Z% over dense-only."*

---

## Configuration knobs (`.env`)

| Variable | Default | What it controls |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedder. |
| `GENERATION_MODEL` | `gpt-4o-mini` | Answer generator. Set to `gpt-4o` for higher quality. |
| `JUDGE_MODEL` | `gpt-4o-mini` | Reranker + verifier + completeness. |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic`. |
| `DEFAULT_CHUNKER` | `recursive` | `fixed_size` / `recursive` / `semantic`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Chunker tuning. |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | `10` / `10` | First-pass retrieval. |
| `RRF_DENSE_WEIGHT` / `RRF_SPARSE_WEIGHT` | `0.7` / `0.3` | Bias toward semantic vs keyword. |
| `RRF_K_CONSTANT` | `60` | RRF dampening constant. |
| `RERANK_INPUT` / `RERANK_OUTPUT` | `20` / `5` | Reranker window. |
| `DEDUP_THRESHOLD` | `0.95` | Cosine threshold for dropping near-duplicates. |
| `CONFIDENCE_THRESHOLD` | `0.35` | Below this → graceful IDK. |

---

## Running the tests

```bash
pytest tests/ -v
```

Tests cover loaders, chunkers, RRF fusion math, deduplication, citation parsing, and BM25 round-trips. They do **not** require an OpenAI API key — every LLM-touching path is exercised through scripts/eval, not unit tests.

---

## Talking points for interviews

1. **"Why hybrid?"** Pure dense retrieval underperforms on technical docs because exact tokens (`SCHEMA_MISMATCH`, `429`, `helix-jit`) don't always cluster well in the embedding space. BM25 catches them. Show the chunking comparison report as evidence.
2. **"What stops the model from hallucinating?"** Three layers: (a) the grounded prompt forbids outside knowledge, (b) the citation verifier flags unsupported citations after generation, (c) the confidence threshold short-circuits to a structured IDK response when retrieval is weak.
3. **"How would you scale this?"** Chroma → Qdrant or pgvector for multi-tenant. BM25 → Elasticsearch / OpenSearch. Reranker → a hosted cross-encoder (e.g., Cohere Rerank, bge-reranker-v2) for lower latency than LLM-as-judge. Cache embeddings by content hash. Move the eval into CI so every PR runs the full suite.
4. **"How do you decide on chunk size?"** Empirically — that's what `compare_chunking.py` is for. Show the table.

---

## License

MIT — do whatever you want.
