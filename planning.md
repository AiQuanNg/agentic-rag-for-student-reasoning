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

### 1.  **Master/Orchestrator Agent**
    -   Splits input into batches (e.g., 10 answers) and dispatches.
    -   Manages retries and confidence thresholds.
    -   Writes run metadata (batch id, timings, outcomes).
### 2.  **Extracting Agent**
**Architecture** (Following Ottomator Pattern):
- **Agent Layer**: Pydantic AI agent with GPT-4o-mini
- **Tools Layer**: 
  - `retrieve_codebook_keywords(question_id)` → Returns approved keywords as strings
  - `search_similar_keywords(term_embedding)` → Optional pgvector search for novel term proximity
- **Validation Layer**: Pydantic models ensure structured outputs

**Responsibilities**:
1. **Keyword Matching** (Semantic, not exact):
   - LLM receives student answer + approved keyword list
   - Matches student concepts to keywords by MEANING (e.g., "survival of fittest" → "natural selection")
   - Novel terms: Flags important concepts NOT in codebook (expansion candidates)

2. **Theme Detection**:
   - Identifies broader subject areas (e.g., "evolutionary biology", "ecological pressure")
   - Detects reasoning approaches (cause-effect, comparative, analogical)
   - Crucial for detecting "Latent" reasoning that uses non-standard frameworks

3. **Evidence Extraction**:
   - Extracts 2-5 key phrases (exact quotes) demonstrating understanding
   - Provides Classifier with concrete evidence spans for reasoning traces

**Output Model** (Pydantic):
```python
class ExtractionResult(BaseModel):
    matched_keywords: List[str] # Keywords from codebook
    detected_themes: List[str]  # Broader reasoning categories
    novel_terms: List[str]      # Codebook expansion candidates
    evidence_spans: List[str]   # Exact quotes from answer
    extraction_confidence: float# 0-1 extraction quality score
    tools_used: List[str]       # Track which retrieval strategies worked
```

**Retrieval Strategy:**

**String-Based**
- Tool returns flat list of approved keywords
- LLM does semantic matching using natural language understanding
- Simpler, clearer attribution for research

**Progressive Parameter Tuning** (3-Attempt Retry):
- Attempt 1: temperature=0.1 (precise extraction)
- Attempt 2: temperature=0.2 (slightly more exploratory)
- Attempt 3: temperature=0.3 (capture edge cases)


### 3.  **Matching / Classifier Agent**
**Purpose**: Classify student answers as Standard / Latent / Off-topic with explainable reasoning

**Architecture** (Hybrid Approach - NOT Pure pgvector):

**Decision Context**: Generic rubrics make pure similarity search ineffective
- Rubric levels describe answer quality (length, logic, evidence)
- Not domain-specific enough for semantic matching
- Low similarity scores across all rubrics for most answers

**Hybrid Classification Strategy**:

1. **Primary: LLM Reasoning with Extractor Context**
   - Input: Student answer + ExtractionResult (keywords, themes, novel_terms, confidence)
   - LLM analyzes using Extractor findings as classification proxy
   - Decision logic encoded in system prompt

2. **Supporting: Rubric Reference (NOT Retrieval)**
   - Tool: `get_question_rubrics()` → Returns ALL 3 levels for reference
   - Tool: `get_classification_criteria()` → Returns Standard vs Latent definitions
   - LLM uses as context, not as similarity-filtered results

3. **Validation: Extractor Integration**
   - matched_keywords count → Indicates Standard terminology usage
   - novel_terms presence → Signals Latent reasoning
   - detected_themes → Reveals reasoning approach
   - extraction_confidence → Proxy for answer clarity

**Tools**:
- `get_question_rubrics(question_id)` → Fetch all rubric levels (no filtering)
- `get_classification_criteria()` → Fetch Standard/Latent definitions

### Output model (Pydantic)
```python
from typing import Literal, List, Dict
from pydantic import BaseModel

class ClassificationResult(BaseModel):
        label: Literal["standard", "latent", "off_topic"]
        classification_confidence: float  # 0-1
        evidence_spans: List[str]         # exact quotes supporting label
        reasoning: str                    # 2-3 sentence explanation
        rubric_alignment: Dict            # which rubric level(s) align
        extractor_context: Dict           # how extraction influenced decision
        criteria_assessment: Dict         # indicators found for Standard vs Latent
```

### Classification decision rules
Use these rules as a transparent decision proxy; the LLM should always include a short reasoning trace.

- STANDARD — any of:
    - `matched_keywords >= 3`
    - direct explanation or explicit themes present
    - clear alignment with the rubric `100` descriptor

- LATENT — all of:
    - `novel_terms > 0`
    - mechanism/analogy themes present
    - understanding demonstrated through non-standard reasoning

- OFF_TOPIC — any of:
    - `matched_keywords < 1`
    - no meaningful themes identified
    - generic platitudes or vague statements only

If none of the above clearly applies, fall back to `classification_confidence` + LLM judgement and include the decision rationale for auditability.

**Why This Works**:
- Leverages already-extracted information (keywords, themes)
- Avoids low-similarity issues with generic rubrics
- Interpretable (decision based on clear signals)
- Flexible (LLM can handle edge cases)

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

### Stage 1: Extractor Retrieval (Keyword Discovery)

**Purpose**: Help LLM see approved vocabulary before extraction

**Method**: String-based retrieval (simple query)

```sql
SELECT keyword FROM topic_keywords
WHERE question_id = $1 AND approved = true
ORDER BY keyword;
```

**LLM receives**: 
- Student answer text
- List of approved keyword strings (e.g., ["artificialintelligence", "creativity", "deeplearning"])

**LLM does**: Semantic matching by meaning, not exact word match

### Stage 2: Classifier Retrieval (Rubric Discovery)

**Purpose**: Provide rubric context without relying on pgvector similarity

**Problem with Pure pgvector Approach**:
- Generic rubric templates ("200-300 words, clear logic") don't vector well
- Low similarity to all rubrics regardless of answer quality
- Can't distinguish between Standard and Latent reasoning

**Adopted Solution: Context-Based Retrieval**

**Method 1: Simple Fetch (Primary)**

-- Get ALL rubrics for question (no filtering)
```sql
SELECT level_pct, descriptor
FROM rubrics
WHERE question_id = $1
ORDER BY level_pct DESC;
```
-- Returns 3 rubrics (100, 50, 0) as context

**Method 2: Extractor-Guided Classification** (Decision Proxy)
```
if extraction_result.matched_keywords >= 3: likely_classification = "standard"
elif extraction_result.novel_terms and "mechanism" in extraction_result.detected_themes: likely_classification = "latent"
else: likely_classification = "off_topic"
```
**Rationale**:
- Extractor already did semantic analysis (keywords, themes)
- Rubrics provide quality expectations (100/50/0 levels)
- LLM synthesizes both → final classification
- No need for complex similarity scoring

**Key Difference**: 
- Extractor: Simple keyword list lookup (LLM matches semantically)
- Classifier: Complex rubric retrieval (pgvector finds best criteria)

------------------------------------------------------------------------

## 7) Off-Topic Handling

-   Maintain `offtopic_signals` (e.g., low similarity to any rubric/keyword; lexicon for unrelated domains).
-   Off-topic still generates a short rationale; store for **misconception analytics** (don't discard).

------------------------------------------------------------------------

## 8) Ingestion Pipelines

-   **Professor inputs**: 4 questions, rubric (0/50/100), criteria, keyword lists.
-   **Student CSVs**: import, normalize, NLTK, embed.
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

