# Development Guidelines for Student Reasoning RAG

### 🔄 Project Awareness & Context

-   **Always read `planning.md`** at the start of a new conversation to understand architecture, agents, and workflow.
-   **Check `task.md`** before starting a new task. If the task isn't listed, add it with today's date.
-   **Follow naming conventions and file structure** from `planning.md` (e.g., `agent/`, `api/`, `cli/`, `sql/`).

------------------------------------------------------------------------

### 🧱 Code Structure & Modularity

-   **Keep files \< 500 lines.** Split into submodules if needed (e.g.,
    `extractor.py`, `classifier.py`).
-   **Separate concerns by feature**: agents, ingestion, DB utilities, API.
-   **Use consistent relative imports** within the package.

------------------------------------------------------------------------

### 🧪 Testing & Reliability

-   **Write Pytest unit tests** for each new feature (agent, ingestion, API endpoint).
-   **Test types**:
    -   1 normal case
    -   1 edge case
    -   1 failure case
-   **Tests live in `/tests`**, mirroring the project structure.
-   **Always run in venv** (`python -m pytest`).
-   Update existing tests when logic changes.

------------------------------------------------------------------------

### 🗄️ Database & Storage

-   **PostgreSQL with pgvector** is the canonical DB.
-   **Schema management**: all changes go through `sql/schema.sql` and migrations (Alembic).
-   **Tables**: `questions`, `rubrics`, `criteria`, `topic_keywords`,
    `student_submissions`, `runs`, `run_items`,
    `classifications`, `reasoning_summaries`, `pattern_proposals`,
    `approvals`, `metrics`.
-   **Confirm migrations** before code references new fields.

------------------------------------------------------------------------

### ✅ Task Completion

-   **Update `TASK_student_reasoning.md`** immediately after completing tasks.
-   Add new subtasks discovered during work under a **"Discovered During Work"** section.

------------------------------------------------------------------------

### 📎 Style & Conventions

-   **Python only**, PEP8 compliant, type hints required.
-   **Format with `black`** and **lint with `ruff`**.
-   **Use Pydantic** for input/output models and validation.
-   **Use FastAPI** for APIs, `asyncpg` or `SQLAlchemy` for DB.
-   **Docstrings required** for every function (Google style).
-   Inline `# Reason:` comments where logic is not obvious.

------------------------------------------------------------------------

### 📚 Documentation & Explainability

-   **Keep `README.md` up to date** (setup, features, usage).
-   **Update `planning.md`** when architecture or workflow evolves.
-   **Comment non-obvious code** to aid mid-level developers.
-   Document evaluation logic (confidence scoring, retry policy).

------------------------------------------------------------------------

### 🧠 AI Behavior Rules

-   **Never assume missing context. Ask clarifying questions.**
-   **Never hallucinate libraries or functions** -- stick to standard or documented dependencies.
-   **Confirm file/module paths** before referencing.
-   **Do not overwrite or delete code** unless explicitly instructed or listed in `task.md`.
