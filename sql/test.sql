SELECT id, keyword, embedding IS NOT NULL AS has_embedding
FROM topic_keywords
WHERE question_id = 1 AND status = 'approved';

SELECT id, level_pct, embedding IS NOT NULL AS has_embedding
FROM rubrics
WHERE question_id = 1;

TRUNCATE TABLE student_submissions RESTART IDENTITY CASCADE;