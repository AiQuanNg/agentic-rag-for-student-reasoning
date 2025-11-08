-- Add constraints separately
ALTER TABLE student_submissions
ADD CONSTRAINT chk_answer_nonempty
CHECK (length(btrim(answer_text)) > 0);

ALTER TABLE rubrics
ADD CONSTRAINT uq_rubrics_question_level
UNIQUE(question_id, level_pct);

ALTER TABLE run_items
ADD CONSTRAINT uq_run_submission
UNIQUE(run_id, submission_id);
