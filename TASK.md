# Task List -- Agentic RAG for Student Reasoning

## Overview

This document tracks all tasks for building the **Agentic RAG system for
student reasoning discovery**. Tasks are organized by phase and
component, following the project plan.

------------------------------------------------------------------------

## Phase 0: Environment & Setup

-   [X] Create a project directory structure
-   [X] Initialize virtual environment (`venv`) and requirements.txt
-   [X] Create `.env.example` with DB + LLM config variables
-   [X] Set up `.gitignore` and pre-commit hooks (ruff, black)

------------------------------------------------------------------------

## Phase 1: Database & Schema

- [x] **PostgreSQL + pgvector schema design** 
  - Import CSVs into all core tables
  - Core tables: questions, rubrics, criteria, topic_keywords
  - Processing tables: runs, run_items, student_submissions
  - Vector indexes with 384-dimensional embeddings

-   [X] Test DB connection
-   [X] Write embedding generation script using `all-MiniLM-L6-v2`
-   [X] Verify embeddings in database


------------------------------------------------------------------------

## Phase 2: Core Agent Development

### Orchestrator & Workflow

-   [X] Implement **Master Orchestrator Agent** to split student answers
    into batches
-   [X] Database integration with asyncpg
-   [X] Write run metadata to DB

### Sub-Agents
### Part A: Extractor Agent (Week 1)

#### Summary
- **Status**: Successfully tested on batch of 10 student answers
- **Model**: OpenRouter Deepseek (free tier) / OpenAI GPT-4o-mini
- **Architecture**: Pydantic AI agent following Ottomator pattern
- **Testing**: CLI tool built and validated

#### Completed Tasks
- [x] Study ottomator-agents layered architecture
- [x] Create folder structure: `agent/tools/`, `agent/prompts/`, `agent/models/`
- [x] Install dependencies: `pydantic-ai`, `rich`, `openai`

**Models (Validation Layer)**:
- [x] `agent/models/extraction.py`:
  - [x] `ExtractionResult` (output model with Pydantic validation)
  - [x] `ExtractorDependencies` (dependency injection)

**Tools Layer**:
- [x] `agent/tools/codebook.py`:
  - [x] `CodebookTools` class with asyncpg pool
  - [x] `get_approved_keywords(question_id)` - String-based retrieval
  - [x] Tool tested with real database

**Prompts Layer**:
- [x] `agent/prompts/extractor.py`:
  - [x] `EXTRACTOR_SYSTEM_PROMPT` (domain-specific for GenAI questions)
  - [x] Few-shot examples (Standard, Latent, Vague answers)
  - [x] Confidence scoring guidelines

**Agent Layer**:
- [x] `agent/extractor.py`:
  - [x] Initialize Pydantic AI agent with configurable provider (OpenRouter/OpenAI)
  - [x] Implement `@extractor_agent.tool` decorator for `retrieve_codebook_keywords`
  - [x] Implement `extract_keywords()` main function
  - [x] Single-attempt approach (retries managed by Orchestrator)

**Configuration**:
- [x] `agent/config/providers.py`:
  - [x] Per-agent provider configuration (EXTRACTOR_PROVIDER, CLASSIFIER_PROVIDER, AGGREGATOR_PROVIDER)
  - [x] Support for OpenRouter (Deepseek free) and OpenAI
  - [x] Easy provider switching via `.env`

**Testing CLI**:
- [x] `cli/test_extractor.py`:
  - [x] Batch processing of 10 answers with progress bar
  - [x] Rich table display of results
  - [x] CSV export for analysis
  - [x] Model info display (provider, model, cost)

#### Lessons Learned
1. **String-based retrieval works well** - LLM handles semantic matching without pgvector
2. **Single-attempt strategy** - Orchestrator will handle retries for low-confidence results
3. **Per-agent provider config** - Enables testing with free models before production
4. **Hybrid approach validated** - Simple tool retrieval + LLM reasoning is effective

### Part B: Classifier Agent  
#### Architecture Decision: Hybrid Approach ✅

**Key Decision**: Classification uses **hybrid strategy**, NOT pure pgvector similarity

**Rationale**:
- Current rubrics are generic templates (work for grading, not semantic retrieval)
- Rubric 100/50/0 describe answer length/logic, not domain concepts
- pgvector similarity would return low scores for all rubrics
- Generic criteria definitions (Standard/Latent) need context-aware interpretation

**Hybrid Classification Strategy**:

1. **Layer 1: Direct LLM Analysis** (Primary)
   - LLM reads student answer
   - LLM analyzes Extractor output (keywords, themes, novel terms)
   - LLM applies classification logic from system prompt

2. **Layer 2: Rubric Context** (Supporting)
   - Retrieve ALL rubrics for question (no filtering)
   - Use as reference context, not similarity-matched results
   - LLM decides which level matches

3. **Layer 3: Extractor Integration** (Decision Proxy)
   - High keyword match → likely STANDARD
   - Novel terms + themes → likely LATENT
   - Low extraction confidence → likely OFF_TOPIC

#### Decision Rules

- **STANDARD** — any of:
  - `matched_keywords >= 3`
  - Direct explanation or explicit themes present
  - Aligns with the rubric "100" descriptor

- **LATENT** — all of:
  - `novel_terms > 0`
  - Mechanism/analogy themes present
  - Understanding demonstrated through non-standard reasoning

- **OFF_TOPIC** — any of:
  - `matched_keywords < 1`
  - No meaningful themes identified
  - Generic platitudes or vague statements only

#### Planned Implementation

**Week Timeline**:

##### Day 1: Models & Architecture
- [ ] Create `agent/models/classification.py`:
  - [ ] `ClassificationResult` (label, confidence, reasoning, evidence)
  - [ ] `ClassifierDependencies` (db_pool, question_id)
  - [ ] `ClassificationInput` (answer + extraction results)

##### Day 2: Tools Layer (Simple Retrieval)
- [ ] Create `agent/tools/rubric.py`:
  - [ ] `get_question_rubrics(question_id)` → Returns ALL 3 levels (0/50/100)
  - [ ] `get_classification_criteria()` → Returns Standard vs Latent definitions
  - [ ] NO similarity search - just fetch all as context

##### Day 3-4: System Prompt & Agent
- [ ] Create `agent/prompts/classifier.py`:
  - [ ] `CLASSIFIER_SYSTEM_PROMPT` with hybrid logic
  - [ ] Decision tree: Use Extractor output to guide classification
  - [ ] Few-shot examples (Standard/Latent/Off-topic with Extractor context)
  - [ ] Tool usage instructions (fetch all rubrics, then reason)

- [ ] Create `agent/classifier.py`:
  - [ ] Initialize Pydantic AI agent with configurable provider
  - [ ] Implement tools: `get_question_rubrics`, `get_classification_criteria`
  - [ ] Implement `classify_answer(answer_text, extraction_results)`
  - [ ] Single-attempt approach (Orchestrator manages retries)

##### Day 5: Testing CLI
- [ ] Create `cli/test_classifier.py`:
  - [ ] Process Extractor output → Classifier
  - [ ] Display classification results with reasoning
  - [ ] Test on same 10 answers used for Extractor
  - [ ] Compare Standard vs Latent vs Off-topic distribution

##### Day 6-7: Validation & Refinement
- [ ] Manual review of 10 test classifications
- [ ] Compare against expected labels (if available)
- [ ] Refine decision rules in system prompt
- [ ] Iterate 2-3 times for accuracy
- [ ] Document results in `CLASSIFIER_VALIDATION.md`

#### Success Criteria
- [ ] 80%+ accuracy on Standard vs Latent distinction (manual review)
- [ ] Clear reasoning provided for each classification
- [ ] Evidence spans extracted correctly
- [ ] Confidence scores calibrated (high confidence = correct label)

####  **Summary Reporter**: 
  -   [ ] Generate per-answer summary

### Aggregator & Professor-in-the-loop
#### Aggregator Development
- [ ] **Novel pattern detection**
  - LLM prompts for theme identification
  - Evidence compilation and ranking
  - Proposal generation for professor review

#### Professor-in-the-loop
- [ ] **Review interface design**
  - Classification result presentation
  - Theme proposal management
  - Approval workflow implementation

- [ ] **Reporting system**
  - Batch summary generation
  - Performance metrics dashboard
  - Export functionality for research

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

## Phase 4: Ingestion System (Drop as manually import csv files in phase 1)

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

-   [ ] End-to-end: ingest → run → classify → summarize → propose keyword
-   [ ] Test confidence + retry cycle
-   [ ] Test professor approval loop

### Metrics

-   [ ] Implement gold set evaluation (precision/recall/F1)
-   [ ] Track confidence distributions, retry rate, approval rate, drift metrics

------------------------------------------------------------------------

## Phase 6: Documentation

-   [ ] Update `README.md` (student reasoning focus)
-   [ ] Maintain `planning.md` and `TASK.md`
-   [ ] Add API documentation with FastAPI `/docs`
-   [ ] Create setup & usage guides (PyCharm + CLI)
-   [ ] Write evaluation guide (metrics, gold set usage)

------------------------------------------------------------------------

## Project Status

### Next Steps:
1. Complete NLTK preprocessing implementation
2. Implement hybrid keyword matching with database integration
3. Create comprehensive unit tests for new components
4. Update orchestrator to use enhanced extractor
