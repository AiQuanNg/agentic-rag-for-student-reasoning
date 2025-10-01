-- sql/alter_embedding_dimension.sql
-- Change embedding dimensions from 1536 to 384 for all-MiniLM-L6-v2

BEGIN;

-- Drop existing vector indexes first
DROP INDEX IF EXISTS idx_questions_embedding;
DROP INDEX IF EXISTS idx_rubrics_embedding;
DROP INDEX IF EXISTS idx_criteria_embedding;
DROP INDEX IF EXISTS idx_keywords_embedding;

-- Alter column types
ALTER TABLE questions ALTER COLUMN embedding TYPE vector(384);
ALTER TABLE rubrics ALTER COLUMN embedding TYPE vector(384);
ALTER TABLE criteria ALTER COLUMN embedding TYPE vector(384);
ALTER TABLE topic_keywords ALTER COLUMN embedding TYPE vector(384);

-- Recreate indexes with new dimension
CREATE INDEX idx_questions_embedding ON questions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_rubrics_embedding ON rubrics USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_criteria_embedding ON criteria USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_keywords_embedding ON topic_keywords USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

COMMIT;