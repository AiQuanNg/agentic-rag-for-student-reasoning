# Task List -- Agentic RAG for Student Reasoning

## Overview

This document tracks all tasks for building the **Agentic RAG system for
student reasoning discovery**. Tasks are organized by phase and
component, following the project plan.

------------------------------------------------------------------------

## Phase 0: Environment & Setup

-   [ ] Create project directory structure
-   [ ] Initialize virtual environment (`venv`) and requirements.txt
-   [ ] Create `.env.example` with DB + LLM config variables
-   [ ] Set up `.gitignore` and pre-commit hooks (ruff, black)

------------------------------------------------------------------------

## Phase 1: Database & Schema

-   [ ] Create PostgreSQL database and enable `pgvector`
-   [ ] Write `schema.sql` with tables: `questions`, `rubrics`,
    `criteria`, `topic_keywords`, `student_submissions`, `chunks`,
    `runs`, `run_items`, `classifications`, `reasoning_summaries`,
    `pattern_proposals`, `approvals`, `metrics`
-   [ ] Create migration scripts with Alembic
-   [ ] Implement DB connection utilities (asyncpg/SQLAlchemy)
-   [ ] Add transaction management and retry logic

------------------------------------------------------------------------

## Phase 2: Core Agent Development

### Orchestrator & Workflow

-   [ ] Implement **Master Orchestrator Agent** to split student answers
    into batches
-   [ ] Add confidence threshold + retry logic (diverse sampling,
    retrieval variations)
-   [ ] Write run metadata to DB

### Sub-Agents

-   [ ] **Extracting Agent**: detect themes/keywords, map to Codebook
-   [ ] **Classifier Agent**: apply rubric + criteria, output
    Standard/Latent/Off-topic
-   [ ] **Summary Reporter**: generate per-answer reasoning summary
-   [ ] **Aggregator Agent**: detect novel patterns, propose codebook
    updates

### Professor-in-the-loop

-   [ ] Build staging area for new keyword/theme proposals
-   [ ] Implement approval/rejection workflow

------------------------------------------------------------------------

## Phase 3: API Layer

-   [ ] Create FastAPI app with lifespan management and logging
-   [ ] Add endpoints:
    -   `POST /runs` → start batch run
    -   `GET /runs/{id}/stream` → SSE progress
    -   `GET /reports/{run_id}` → consolidated results
    -   `POST /proposals/{id}/approve|reject` → professor review
    -   `GET /health` → health check
-   [ ] Implement error handling + middleware (CORS, logging)

------------------------------------------------------------------------

## Phase 4: Ingestion System

-   [ ] CSV importers for professor inputs (questions, rubric, criteria,
    keywords)
-   [ ] CSV importer for student answers
-   [ ] Semantic chunking of long answers
-   [ ] Embedding generation and storage in `chunks` table
-   [ ] Backfill mechanism when codebook updates are approved

------------------------------------------------------------------------

## Phase 5: Evaluation & Testing

### Unit Tests

-   [ ] Test orchestrator (batch splitting, retries)
-   [ ] Test sub-agents independently
-   [ ] Test DB ingestion and schema constraints
-   [ ] Test API endpoints with mocks

### Integration Tests

-   [ ] End-to-end: ingest → run → classify → summarize → propose
    keyword
-   [ ] Test confidence + retry cycle
-   [ ] Test professor approval loop

### Metrics

-   [ ] Implement gold set evaluation (precision/recall/F1)
-   [ ] Track confidence distributions, retry rate, approval rate, drift
    metrics

------------------------------------------------------------------------

## Phase 6: Documentation

-   [ ] Update `README.md` (student reasoning focus)
-   [ ] Maintain `planning.md` and `TASK.md`
-   [ ] Add API documentation with FastAPI `/docs`
-   [ ] Create setup & usage guides (PyCharm + CLI)
-   [ ] Write evaluation guide (metrics, gold set usage)

------------------------------------------------------------------------

## Phase 7: CLI & Reporting

-   [ ] Implement CLI for ingestion, run, evaluation\
-   [ ] Add options: `ingest`, `run`, `eval`, `export`\
-   [ ] Support streaming output in terminal\
-   [ ] Export reports (CSV/JSON)

------------------------------------------------------------------------

## Phase 8: Quality Assurance

-   [ ] Linting (ruff), formatting (black), typing (mypy)\
-   [ ] Code review for best practices and security\
-   [ ] Validate schema migrations\
-   [ ] Performance test batch scaling (1000+ answers)\
-   [ ] Review data governance (PII handling, anonymization)

------------------------------------------------------------------------

## Project Status

🚧 **In Progress**\
- Planning complete (`planning.md`)\
- README drafted (`README.md`)\
- Next step: implement **schema.sql** and ingestion pipeline
