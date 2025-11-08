# Data Dictionary

## `questions`

- **Rows**: 5  
- **Columns**: id, code, text, metadata

- **Primary Key**: id


| Column     | Type   | Unique values | Example(s)                                                            | Constraints / Description                                     |
|------------|--------|--------------:|-----------------------------------------------------------------------|---------------------------------------------------------------|
| `id`       | INT    |             4 | 1, 2, 3, 4                                                            | PK                                                            |
| `code`     | string |             4 | Q1, Q2, Q3, Q4                                                        | Short code                                                    |
| `text`     | string |             4 | What is generative AI and how does it work?                           | Full question text (prompt shown to students).                |
| `metadata` | JSONB  |             4 | {"goal": "Learn about generative AI basics", "topic": "fundamentals"} | JSON (string); JSON or free-form metadata about the question. |


## `criteria`

- **Rows**: 2  
- **Columns**: id, question_id, name, description, guidance

- **Primary Key**: id

- **Foreign Key**: `question_id` → `questions.id`


| Column        | Type   | Unique values | Example(s)                                                                                                                            | Constraints / Description                                |
|---------------|--------|--------------:|---------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `id`          | INT    |             2 | 1, 2                                                                                                                                  | PK.                                                      |
| `name`        | string |             2 | Standard Answer, Latent Answer                                                                                                        | Short label or title for a criterion.                    |
| `description` | string |             2 | Meets rubric fully with accurate, clear but surface-level reasoning, Meets rubric fully with deeper reasoning and critical engagement | Narrative description of the criterion.                  |
| `guidance`    | string |             2 | Comprehension & Recall: The answer accurately Restates facts, definitions, or examples directly from materials or online.             | Evaluator guidance and notes for applying the criterion. |


## `topic_keywords`

- **Rows**: 671  
- **Columns**: id, question_id, keyword, status, source

- **Primary Key**: id

- **Foreign Key**: `question_id` → `questions.id`

- **Unique Together**: (question_id, keyword)


| Column        | Type   | Example(s)                 | Constraints / Description                                                                      |
|---------------|--------|----------------------------|------------------------------------------------------------------------------------------------|
| `id`          | INT    | 1, 2, 3, 4                 | PK                                                                                             |
| `question_id` | INT    | 1, 2, 3, 4                 | FK; Reference to questions.id (FK).                                                            |
| `keyword`     | string | adversarial, ai, algorithm | Keyword/term linked to a question for evaluation or retrieval.                                 |
| `status`      | string | approved                   | Allowed: approved/pending/rejected; Status of the keyword (e.g., approved, pending, rejected). |
| `source`      | string | professor                  | Origin of the keyword (e.g., professor, online).                                               |




## `rubrics`

- **Rows**: 12  
- **Columns**: id, question_id, level_pct, descriptor, exemplar, Unnamed: 5

- **Primary Key**: id

- **Foreign Key**: `question_id` → `questions.id`

- **Notes**: Uses level_pct instead of (score_min, score_max).


| Column        | Type   | Unique values | Example(s)                                                                                                                | Constraints / Description                                     |
|---------------|--------|--------------:|---------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `id`          | INT    |            12 | 1, 2, 3, 4                                                                                                                | PK.                                                           |
| `question_id` | INT    |             4 | 1, 2, 3, 4                                                                                                                | FK; Reference to questions.id (FK).                           |
| `level_pct`   | INT    |             3 | 100, 50, 0                                                                                                                | Rubric score level as percent (0,50,100).                     |
| `descriptor`  | string |             3 | Full understanding, Partial understanding, No understanding                                                               | Level descriptor (what performance looks like at this level). |
| `exemplar`    | string |             6 | Answer is 200–300 words, uses clear logic, clearly defines generative AI an explains how it works, and provides evidence, | Example answer or evidence illustrating the level.            |


