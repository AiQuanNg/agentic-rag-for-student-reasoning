# Project Plan --- Agentic RAG for Student Reasoning

## 1) Goals & Success Criteria

**Primary goal:** Classify each student answer as **Standard / Latent /
Off-topic**, explain the reasoning, summarize themes, and continuously
refine a **Thematic Codebook** with professor oversight.\
**Success looks like:** 
- ≥ defined precision/recall on classification against a held-out, professor labeled set.
- Clear, auditable **reasoning traces** per answer. 
- Stable, professor-approved updates to keywords/themes without noisy drift. 
- Scalable batch processing (1→N answers) with reproducible runs.

------------------------------------------------------------------------

## 2) Core User Stories

-   **Instructor** reviews per-answer classifications + reasoning +
    theme summary; approves/edits proposed new keywords.
-   **Researcher** exports aggregated patterns across
    assignments/batches.
-   **Operator** triggers batch runs, monitors confidence, retrigger
    logic, and health metrics.

------------------------------------------------------------------------

## 3) System Architecture

-   **API Layer:** FastAPI (SSE streaming for long tasks).
-   **Agent Layer:** Orchestrator + Sub-agents (Extractor, Matcher/Classifier, Aggregator, Summary Reporter).
-   **Storage:** PostgreSQL (+ pgvector) for docs, embeddings, runs, and codebook; optional object storage for raw files.
-   **Orchestration:** Batch runner (CLI + API); async jobs via lightweight queue (e.g., asyncio tasks / RQ / Celery---pick one).
-   **Observability:** Structured logs, run metadata, confidence scores, drift metrics.

------------------------------------------------------------------------

## 4) Agents & Responsibilities

1.  **Master/Orchestrator Agent**
    -   Splits input into batches (e.g., 10 answers) and dispatches.
    -   Manages retries and confidence thresholds.
    -   Writes run metadata (batch id, timings, outcomes).
2.  **Extracting Agent**
    -   Detects **themes/sentiment**, matches to **Codebook keywords**.
    -   Emits: matched_keywords, detected_themes, salient evidence spans.
3.  **Matching / Classifier Agent**
    -   Applies rubric + criteria to classify **Standard / Latent / Off-topic**.
    -   Produces: label, evidence, rubric alignment score, confidence.
4.  **Summary Reporter (Phase 2)**
    -   Generates **per-answer summary** (classification + reasoning + theme note).
    -   Clears short-term memory after each batch; persists outputs.
5.  **Aggregator Agent**
    -   Detects **novel patterns** if the answer did not meet the confidence threshold 3 times in a row.
    -   Proposes **codebook updates** (new keywords/aliases, theme candidates) with frequency stats.
6.  **Professor-in-the-loop**
    -   Approves/edits proposed keywords/themes in a **staging** table.
    -   Approved items move to **Codebook (long-term)** and become available to the next run.

------------------------------------------------------------------------

## 5) Data Model (PostgreSQL + pgvector)

**Entities (tables)** 
- `questions` --- 4 questions (id, code, text, metadata). 
- `rubrics` --- id, question_id, level_pct (0/50/100), descriptors, exemplars, embedding. 
- `criteria` --- criteria for **Standard vs Latent** (id, name, description, guidance, embedding). 
- `topic_keywords` --- professor keyword sets (topic_id, keyword, weight, status: approved/staged, source).
- `classifications` --- final label, rubric alignment score, evidence spans. 
- `reasoning_summaries` --- concise rationale + theme summary per answer. 
- `pattern_proposals` --- aggregator's proposed new keywords/themes (freq, examples, status). 
- `approvals` --- professor decisions (approve/reject/edit with comment and editor id). 
- `metrics` --- offline eval (precision/recall/F1 by label, drift, approval rate). 

> Vectorization: 
- `rubrics.embedding`, `criteria.embedding`, `topic_keywords.embedding` using `pgvector`. 
- Indexes: `ivfflat` on embeddings, plus btree on foreign keys and timestamps.

------------------------------------------------------------------------

## 6) Retrieval & Reasoning Strategy

-   **Hybrid retrieval**:
    1)  Keyword match (approved `topic_keywords`)\
    2)  Semantic similarity (pgvector on `chunks`/`rubrics`)\
    3)  Rule overlay from `criteria` for edge cases.
-   **Confidence** = weighted blend:
    -   classifier probability (from model output or calibrated score),
    -   rubric alignment score,
    -   agreement across N diverse inference passes (majority vote).
-   **Retry policy**:
    -   On low confidence, perform:
        -   different retrieval slices (top-k variations),
        -   temperature/decoding variation,
        -   structured prompt variant (rubric-first vs evidence-first).
    -   Stop after K diverse tries; mark **needs-review** if still low.

------------------------------------------------------------------------

## 7) Off-Topic Handling

-   Maintain `offtopic_signals` (e.g., low similarity to any rubric/keyword; lexicon for unrelated domains).
-   Off-topic still generates a short rationale; store for **misconception analytics** (don't discard).

------------------------------------------------------------------------

## 8) Ingestion Pipelines

-   **Professor inputs**: 4 questions, rubric (0/50/100), criteria, keyword lists.
-   **Student CSVs**: import, normalize, chunk (if long), embed.
-   **Keyword updates**: proposals → staging → professor approval → activation.
-   **Backfill**: re-run past submissions automatically when codebook updates activate (versioned codebook).

------------------------------------------------------------------------

## 9) API & CLI

-   **FastAPI endpoints**
    -   `POST /runs` start batch with config (question_ids, size, thresholds).
    -   `GET /runs/{id}/stream` SSE progress (per-answer events).
    -   `GET /reports/{run_id}` consolidated JSON/CSV export.
    -   `POST /proposals/{id}/approve|reject` (professor role).
-   **CLI**
    -   `python cli.py ingest --file answers.csv`
    -   `python cli.py run --question 1 --batch-size 10 --threshold 0.8`
    -   `python cli.py eval --run-id ...`

------------------------------------------------------------------------

## 10) Evaluation & QA

-   **Gold set**: 50--100 labeled answers covering each class; maintain versioned labels.
-   **Metrics dashboard**:
    -   Precision/Recall/F1 per class,
    -   Avg confidence, retry rate, approval rate,
    -   Keyword drift (new-keyword proposal rate and acceptance).
-   **Human review loop**:
    -   Surface low-confidence or disagreement cases first.

------------------------------------------------------------------------

## 11) Observability & Governance

-   **Structured logs**: per agent step, retrieval context hashes, model version.
-   **Run registry**: exact prompts, parameters, seeds, embeddings version.
-   **Data protection**: PII minimization, opt-out flags, access controls for professor vs operator.
-   **Safety checks**: profanity/offensive content filter; redact in outputs where necessary.

------------------------------------------------------------------------

## 12) Risks & Mitigations

-   **Professor bottleneck** → Batch approval UI + frequency thresholds for proposals.
-   **Concept drift** → Scheduled evaluation; alert on metric regression.
-   **Noisy keywords** → Require min frequency + cross-question occurrence before proposing.
-   **Over-retrying** → Cap retries; escalate to human review.

------------------------------------------------------------------------

## 13) Implementation Phases (sequence, no time promises)

**Phase A—Foundation** 
- Project skeleton (Poetry/venv), env files, logging. 
- PostgreSQL + pgvector setup; schemas. 
- Embedding service and DB utilities.

**Phase B --- Ingestion & Data Model** 
- Importers for questions/rubrics/criteria/keywords and student CSVs. 
- Embedding + chunking jobs; indexing.

**Phase C—Agents** 
- Orchestrator + Extracting + Classifier + Summary Reporter. 
- Confidence and retry policy; write traces to DB.

**Phase D—Aggregator & Professor Loop** 
- Pattern mining + proposal staging. 
- Approval endpoints + simple review UI (could be minimal FastAPI pages).

**Phase E—API/CLI & Reporting** 
- Batch run endpoints (SSE), exports(CSV/JSON). 
- Run registry and reproducibility metadata.

**Phase F—Evaluation & Hardening** 
- Gold set + automated evaluation scripts. 
- Metrics dashboard; alerts on regressions; security pass.

------------------------------------------------------------------------

## 14) Deliverables Checklist

-   [ ] SQL schema + migrations (pgvector ready).
-   [ ] Ingestion scripts for professor inputs & student CSVs.
-   [ ] Agent implementations with tests.
-   [ ] Confidence/Retry logic with configuration.
-   [ ] Aggregator + proposals staging & approval flow.
-   [ ] FastAPI endpoints + minimal UI for approvals.
-   [ ] CLI tools for ingest/run/eval.
-   [ ] Evaluation suite + metrics export.
-   [ ] Documentation: setup, ops runbook, data governance.

------------------------------------------------------------------------

## 15) PyCharm-Friendly Setup Notes

-   Create virtualenv, install `requirements.txt`, configure
    **Run/Debug**:
    -   `uvicorn app.api:app --reload`
    -   environment vars via `.env`

