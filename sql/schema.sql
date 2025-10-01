
-- sql/schema.sql
-- Agentic RAG for Student Reasoning
-- PostgresSQL + pgvector schema
-- Run with: psql -d agentic_rag -f sql/schema.sql

BEGIN;

-- 0) Extensions ---------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector

-- 1) Enums -------------------------------------------------------------------
DO $$ BEGIN
    CREATE TYPE label_enum AS ENUM ('standard','latent','off-topic');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE keyword_status_enum AS ENUM ('approved','staged','rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE proposal_status_enum AS ENUM ('staged','approved','rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 2) Tables ------------------------------------------------------------------

-- 2.1 questions
CREATE TABLE IF NOT EXISTS questions (
    id           BIGSERIAL PRIMARY KEY,
    code         TEXT UNIQUE,
    text         TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding    vector(1536)  -- adjust to your embedding model
);

COMMENT ON TABLE questions IS 'Canonical prompts / questions';
COMMENT ON COLUMN questions.embedding IS 'Semantic vector of question text (optional)';

-- 2.2 rubrics
CREATE TABLE IF NOT EXISTS rubrics (
    id           BIGSERIAL PRIMARY KEY,
    question_id  BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    level_pct    INT NOT NULL CHECK (level_pct IN (0,5,10,50,100)),
    exemplar     TEXT NOT NULL,
    descriptor   TEXT,
    embedding    vector(1536)
);

-- Exactly one row per band per question
CREATE UNIQUE INDEX IF NOT EXISTS uq_rubrics_question_level
    ON rubrics(question_id, level_pct);

-- 2.3 criteria
CREATE TABLE IF NOT EXISTS criteria (
    id           BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT,
    guidance     TEXT,
    embedding    vector(1536)
);

-- 2.4 topic_keywords
CREATE TABLE IF NOT EXISTS topic_keywords (
    id           BIGSERIAL PRIMARY KEY,
    question_id  BIGINT REFERENCES questions(id) ON DELETE SET NULL,
    keyword      TEXT NOT NULL,
    status       keyword_status_enum NOT NULL,
    source       TEXT NOT NULL,
    embedding    vector(1536)
);

-- Global + per-question namespace uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS uq_topic_keywords_qid_keyword
    ON topic_keywords(question_id, lower(keyword));

CREATE INDEX IF NOT EXISTS idx_topic_keywords_qid_kw
    ON topic_keywords(question_id, lower(keyword));

-- 3) Processing Outputs ------------------------------------------------------

-- 3.1 classifications
CREATE TABLE IF NOT EXISTS classifications (
    id             BIGSERIAL PRIMARY KEY,
    answer_id      TEXT NOT NULL,
    question_id    BIGINT NOT NULL REFERENCES questions(id) ON DELETE RESTRICT,
    label          label_enum NOT NULL,
    rubric_score   NUMERIC(5,2),
    evidence       JSONB NOT NULL DEFAULT '[]'::jsonb,
    rationale      TEXT,
    batch_id       TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_class_answer ON classifications(answer_id);
CREATE INDEX IF NOT EXISTS idx_class_qid ON classifications(question_id);
CREATE INDEX IF NOT EXISTS idx_class_batch ON classifications(batch_id);

-- Optional uniqueness (enable if you want one classification per answer+question)
-- CREATE UNIQUE INDEX uq_class_answer_question ON classifications(answer_id, question_id);

-- 3.2 reasoning_summaries
CREATE TABLE IF NOT EXISTS reasoning_summaries (
    id             BIGSERIAL PRIMARY KEY,
    answer_id      TEXT NOT NULL,
    question_id    BIGINT NOT NULL REFERENCES questions(id) ON DELETE RESTRICT,
    summary        TEXT NOT NULL,
    batch_id       TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sum_answer ON reasoning_summaries(answer_id);
CREATE INDEX IF NOT EXISTS idx_sum_qid ON reasoning_summaries(question_id);
CREATE INDEX IF NOT EXISTS idx_sum_batch ON reasoning_summaries(batch_id);

-- 3.3 pattern_proposals
CREATE TABLE IF NOT EXISTS pattern_proposals (
    id           BIGSERIAL PRIMARY KEY,
    question_id  BIGINT REFERENCES questions(id) ON DELETE SET NULL,
    keyword      TEXT,
    theme        TEXT,
    frequency    INT NOT NULL DEFAULT 0,
    examples     JSONB NOT NULL DEFAULT '[]'::jsonb,
    status       proposal_status_enum NOT NULL DEFAULT 'staged',
    created_by   TEXT NOT NULL DEFAULT 'aggregator',
    batch_id     TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_prop_qid ON pattern_proposals(question_id);
CREATE INDEX IF NOT EXISTS idx_prop_status ON pattern_proposals(status);
CREATE INDEX IF NOT EXISTS idx_prop_batch ON pattern_proposals(batch_id);

-- 4) Approvals & Metrics (from TASK.md) -------------------------------------

-- 4.1 approvals
CREATE TABLE IF NOT EXISTS approvals (
    id             BIGSERIAL PRIMARY KEY,
    proposal_id    BIGINT NOT NULL REFERENCES pattern_proposals(id) ON DELETE CASCADE,
    action         proposal_status_enum NOT NULL CHECK (action IN ('approved','rejected')),
    editor_id      TEXT NOT NULL,
    comment        TEXT,
    decided_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_approvals_proposal ON approvals(proposal_id);
CREATE INDEX IF NOT EXISTS idx_approvals_action ON approvals(action);

-- 4.2 metrics
CREATE TABLE IF NOT EXISTS metrics (
    id               BIGSERIAL PRIMARY KEY,
    run_id           TEXT,
    label            label_enum,
    precision        NUMERIC(6,4),
    recall           NUMERIC(6,4),
    f1               NUMERIC(6,4),
    support          INT,
    retry_rate       NUMERIC(6,4),
    avg_confidence   NUMERIC(6,4),
    approval_rate    NUMERIC(6,4),
    drift_score      NUMERIC(6,4),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_label ON metrics(label);

-- 5) Optional Vector Indexes (enable if using embeddings) --------------------

-- Questions embedding index (cosine distance)
DO $$ BEGIN
    PERFORM 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_questions_embedding';
    IF NOT FOUND THEN
        EXECUTE 'CREATE INDEX idx_questions_embedding ON questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
    END IF;
END $$;

-- Rubrics embedding index
DO $$ BEGIN
    PERFORM 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_rubrics_embedding';
    IF NOT FOUND THEN
        EXECUTE 'CREATE INDEX idx_rubrics_embedding ON rubrics USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
    END IF;
END $$;

-- Criteria for embedding index
DO $$ BEGIN
    PERFORM 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_criteria_embedding';
    IF NOT FOUND THEN
        EXECUTE 'CREATE INDEX idx_criteria_embedding ON criteria USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
    END IF;
END $$;

-- Topic keywords embedding index
DO $$ BEGIN
    PERFORM 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_keywords_embedding';
    IF NOT FOUND THEN
        EXECUTE 'CREATE INDEX idx_keywords_embedding ON topic_keywords USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);';
    END IF;
END $$;

COMMIT;

-- End of schema.sql
