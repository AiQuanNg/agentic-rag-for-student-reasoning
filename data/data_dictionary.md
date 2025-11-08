# Data Dictionary — Agentic RAG

This document describes the long-term memory and processing output tables for the Agentic RAG system, as well as the CSV input file format for student answers.

---

# Long-Term Memory Tables (Database)

## 1) questions
Canonical prompts.

**Columns**
- `id` BIGSERIAL PK
- `code` TEXT UNIQUE — e.g., Q1…Q4
- `text` TEXT NOT NULL — full question
- `metadata` JSONB DEFAULT '{}' — optional tags (goal/topic/difficulty)
- `embedding` vector(1536) (optional) — semantic vector of `text`

**Indexes**
- (optional) `idx_questions_embedding` IVFFLAT on `embedding` (cosine)

**Notes**
- Keep `embedding` if you want semantic checks (off-topic detection, clustering).

---

## 2) rubrics
Per-question scoring bands (0/5/10) with content-specific exemplars.

**Columns**
- `id` BIGSERIAL PK
- `question_id` BIGINT NOT NULL → `questions(id)` ON DELETE CASCADE
- `level_pct` INT NOT NULL — allowed {0,5,10} (or {0,50,100} if you prefer; be consistent)
- `exemplar` TEXT NOT NULL — what this level looks like
- `descriptor` TEXT NULL — optional short label (Full/Partial/None)
- `embedding` vector(1536) (optional) — embedding of `exemplar`

**Constraints**
- UNIQUE (`question_id`, `level_pct`) — exactly one row per band per question

**Indexes**
- (if using embeddings) `idx_rubrics_embedding` IVFFLAT on `embedding` (cosine)

**Why embed?**
- Lets you rank rubric rows by semantic similarity to an answer vector to stabilize scoring/explanations.

---

## 3) criteria
Cross-question reasoning depth (Standard vs Latent).

**Columns**
- `id` BIGSERIAL PK
- `question_id` BIGINT NULL → `questions(id)` ON DELETE SET NULL  
  (NULL = global criteria; you can add per-question overrides later)
- `name` TEXT NOT NULL — e.g., Standard Answer, Latent Answer
- `description` TEXT NULL — concise definition
- `guidance` TEXT NULL — operational cues for the agent/evaluator
- `embedding` vector(1536) (optional) — embedding of `name || ': ' || guidance`

**Indexes**
- (if using embeddings) `idx_criteria_embedding` IVFFLAT (cosine)

**Why embed?**
- Lets the classifier compare an answer vector to criteria vectors to differentiate what vs why/how reasoning.

---

## 4) topic_keywords
Simplified codebook for retrieval and matching.

**Columns**
- `id` BIGSERIAL PK
- `question_id` BIGINT NULL → `questions(id)` ON DELETE SET NULL  
  (NULL = global keyword for all questions)
- `keyword` TEXT NOT NULL — normalized (lowercase, trimmed; optional snake_case)
- `status` TEXT NOT NULL — allowed: approved | staged | rejected
- `source` TEXT NOT NULL — e.g., professor, aggregator, import
- `embedding` vector(1536) (optional) — embedding of `keyword`

**Constraints**
- UNIQUE (`question_id`, `keyword`) — NULL `question_id` forms a global namespace

**Indexes**
- (if using embeddings) `idx_keywords_embedding` IVFFLAT (cosine)
- Helper: `idx_keywords_qid_kw` on (`question_id`, `keyword`)

**Why embed?**
- Improves recall (synonyms/typos) when combined with exact match + filters.

---

# Processing Outputs (Database)

## 5) classifications
Final label & score per processed answer.

**Columns**
- `id` BIGSERIAL PK
- `answer_id` TEXT NOT NULL — external key from CSV (UUID/string)
- `question_id` BIGINT NOT NULL → `questions(id)`
- `label` TEXT NOT NULL — allowed: standard | latent | offtopic
- `rubric_score` NUMERIC(5,2) NULL — your 0/5/10 (or 0–100) score
- `evidence` JSONB DEFAULT '[]' — array of quoted spans/notes
- `rationale` TEXT NULL — concise explanation
- `batch_id` TEXT NULL — optional group/run identifier
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT now()

**Indexes**
- `idx_class_answer` on (answer_id)
- `idx_class_qid` on (question_id)
- (optional) `idx_class_batch` on (batch_id)

**Notes**
- Enforce one classification per (answer_id, question_id, batch_id) if you want batch versions; otherwise (answer_id, question_id) unique.

---

## 6) reasoning_summaries
Short, instructor-friendly summary per answer.

**Columns**
- `id` BIGSERIAL PK
- `answer_id` TEXT NOT NULL — external key from CSV
- `question_id` BIGINT NOT NULL → `questions(id)`
- `summary` TEXT NOT NULL
- `batch_id` TEXT NULL
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT now()

**Indexes**
- `idx_sum_answer` on (answer_id)
- `idx_sum_qid` on (question_id)
- (optional) `idx_sum_batch` on (batch_id)

**Notes**
- If you want strict 1:1 with classification, add UNIQUE (answer_id, question_id, batch_id).

---

## 7) pattern_proposals
Aggregator’s candidate new keywords/themes for professor review.

**Columns**
- `id` BIGSERIAL PK
- `question_id` BIGINT NULL → `questions(id)`
- `keyword` TEXT NULL
- `theme` TEXT NULL
- `frequency` INT DEFAULT 0 — count across the processed batch
- `examples` JSONB DEFAULT '[]' — short excerpts supporting the proposal
- `status` TEXT NOT NULL DEFAULT 'staged' — allowed: staged | approved | rejected
- `created_by` TEXT NOT NULL DEFAULT 'aggregator'
- `batch_id` TEXT NULL
- `created_at` TIMESTAMPTZ NOT NULL DEFAULT now()

**Indexes**
- `idx_prop_qid` on (question_id)
- `idx_prop_status` on (status)
- (optional) `idx_prop_batch` on (batch_id)

**Notes**
- When approved, insert into topic_keywords with status='approved'.

---

# Input (CSV Only, Not in Database)

## student_answers.csv
Ingested by Master Agent in batches of 10.

**Columns**
- `answer_id` — stable external ID (string/UUID); unique per row
- `question_id` — 1..4, must match questions.id
- `student_id` — pseudonymous id (optional, for reporting)
- `answer_text` — raw answer
- `metadata` — JSON string (optional: section/cohort)

**Notes**
- Master Agent embeds answer_text, orchestrates processing, and writes outputs into classifications, reasoning_summaries, and pattern_proposals using answer_id/question_id/batch_id as keys.
