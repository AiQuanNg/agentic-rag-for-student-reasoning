-- Agentic RAG for Student Reasoning
-- Clean PostgreSQL + pgvector schema
-- Run with: psql -d agentic_rag -f schema.sql

-- 0) Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1) Enums
DO $$ BEGIN
CREATE TYPE label_enum AS ENUM ('standard','latent','off-topic');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
CREATE TYPE keyword_status_enum AS ENUM ('approved','staged','rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
CREATE TYPE proposal_status_enum AS ENUM ('staged','approved','rejected');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- 2) Core Tables
CREATE TABLE IF NOT EXISTS questions (
  id BIGSERIAL PRIMARY KEY,
  code TEXT UNIQUE,
  text TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding vector(384)
);

CREATE TABLE IF NOT EXISTS rubrics (
  id BIGSERIAL PRIMARY KEY,
  question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
  level_pct INT NOT NULL CHECK (level_pct IN (0,5,10,50,100)),
  exemplar TEXT NOT NULL,
  descriptor TEXT,
  embedding vector(384)
);

CREATE TABLE IF NOT EXISTS criteria (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  guidance TEXT,
  embedding vector(384)
);

CREATE TABLE IF NOT EXISTS topic_keywords (
  id BIGSERIAL PRIMARY KEY,
  question_id BIGINT REFERENCES questions(id) ON DELETE SET NULL,
  keyword TEXT NOT NULL,
  status keyword_status_enum NOT NULL,
  source TEXT NOT NULL,
  embedding vector(384)
);

CREATE TABLE IF NOT EXISTS student_submissions (
  id BIGSERIAL PRIMARY KEY,
  student_id TEXT NOT NULL,
  question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE RESTRICT,
  answer_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  question_id BIGINT,
  total_items INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'created',
  config JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS run_items (
  id BIGSERIAL PRIMARY KEY,
  run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  submission_id BIGINT NOT NULL REFERENCES student_submissions(id) ON DELETE RESTRICT,
  status TEXT NOT NULL DEFAULT 'queued',
  attempt_count INTEGER NOT NULL DEFAULT 0,
  result JSONB,
  error TEXT,
  confidence_history JSONB DEFAULT '[]'::jsonb,
  needs_aggregator BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3) Output Tables
CREATE TABLE IF NOT EXISTS classifications (
  id BIGSERIAL PRIMARY KEY,
  answer_id TEXT NOT NULL,
  question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE RESTRICT,
  label label_enum NOT NULL,
  rubric_score NUMERIC(5,2),
  evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
  rationale TEXT,
  batch_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reasoning_summaries (
  id BIGSERIAL PRIMARY KEY,
  answer_id TEXT NOT NULL,
  question_id BIGINT NOT NULL REFERENCES questions(id) ON DELETE RESTRICT,
  summary TEXT NOT NULL,
  batch_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pattern_proposals (
  id BIGSERIAL PRIMARY KEY,
  question_id BIGINT REFERENCES questions(id) ON DELETE SET NULL,
  keyword TEXT,
  theme TEXT,
  frequency INT NOT NULL DEFAULT 0,
  examples JSONB NOT NULL DEFAULT '[]'::jsonb,
  status proposal_status_enum NOT NULL DEFAULT 'staged',
  created_by TEXT NOT NULL DEFAULT 'aggregator',
  batch_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS approvals (
  id BIGSERIAL PRIMARY KEY,
  proposal_id BIGINT NOT NULL REFERENCES pattern_proposals(id) ON DELETE CASCADE,
  action proposal_status_enum NOT NULL,
  editor_id TEXT NOT NULL,
  comment TEXT,
  decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS metrics (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT,
  label label_enum,
  precision NUMERIC(6,4),
  recall NUMERIC(6,4),
  f1 NUMERIC(6,4),
  support INT,
  retry_rate NUMERIC(6,4),
  avg_confidence NUMERIC(6,4),
  approval_rate NUMERIC(6,4),
  drift_score NUMERIC(6,4),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4) Basic Indexes
CREATE INDEX IF NOT EXISTS idx_student_submissions_question_id ON student_submissions(question_id);
CREATE INDEX IF NOT EXISTS idx_run_items_run_status ON run_items(run_id, status);
CREATE INDEX IF NOT EXISTS idx_classifications_answer ON classifications(answer_id);
CREATE INDEX IF NOT EXISTS idx_classifications_question ON classifications(question_id);

-- 5) Vector Indexes
DROP INDEX IF EXISTS idx_questions_embedding;
DROP INDEX IF EXISTS idx_rubrics_embedding;
DROP INDEX IF EXISTS idx_criteria_embedding;
DROP INDEX IF EXISTS idx_keywords_embedding;

CREATE INDEX idx_questions_embedding ON questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_rubrics_embedding ON rubrics USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_criteria_embedding ON criteria USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_keywords_embedding ON topic_keywords USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
