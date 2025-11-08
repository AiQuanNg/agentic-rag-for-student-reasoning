-- Import questions data
-- Run with: psql -U rag -d agentic_rag -f sql/import_questions.sql

-- Clear existing data (optional, remove if you want to keep existing data)
-- TRUNCATE TABLE questions CASCADE;

-- Insert questions with proper JSONB formatting
INSERT INTO questions (code, text, metadata) VALUES
('Q1', 
 'What is generative AI and how does it work?',
 '{"goal": "Learn about generative AI basics", "topic": "fundamentals"}'::jsonb),

('Q2', 
 'Can generative AI be used in an innovative way by organizations for strategic purposes, such as to reduce product/service cost, improve customer service, improve employee productivity, etc.? Why or why not? And how? Please provide illustrative examples to support your arguments.',
 '{"goal": "Explore strategic uses of generative AI", "topic": "application"}'::jsonb),

('Q3', 
 'What are the legal and ethical issues associated with generative AI in organizational settings?',
 '{"goal": "Understand risks of generative AI", "topic": "legal_ethics"}'::jsonb),

('Q4', 
 'What are the key challenges when applying generative AI to an innovative idea for achieving strategic benefits for an organization?',
 '{"goal": "Evaluate practical challenges of AI adoption", "topic": "challenges"}'::jsonb)

ON CONFLICT (code) DO UPDATE SET
    text = EXCLUDED.text,
    metadata = EXCLUDED.metadata;

-- Verify the import
SELECT id, code, metadata->>'goal' as goal, metadata->>'topic' as topic 
FROM questions 
ORDER BY id;
