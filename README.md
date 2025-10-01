# Agentic RAG for Student Reasoning

An agent-based RAG system for analyzing student answers, classifying them as **Standard**, **Latent**, or **Off-topic**, and refining a
**Thematic Codebook** with professor oversight.\
This combines **retrieval-augmented generation (RAG)**, **rubric-based classification**, and **professor-in-the-loop feedback** for adaptive,
explainable evaluation.

------------------------------------------------------------------------

## ✨ Features

-   **Multi-agent workflow** (Orchestrator, Extractor, Classifier, Summary Reporter, Aggregator)
-   **Rubric & Criteria-driven classification** (Standard / Latent / Off-topic)
-   **Professor-in-the-loop approvals** for new themes/keywords
-   **PostgreSQL + pgvector** for semantic retrieval
-   **Sentence-Transformers** (`all-MiniLM-L6-v2`) for efficient, local embeddings
-   **Batch processing** of student submissions with retries & confidence thresholds
-   **Evaluation dashboard** for precision/recall and drift detection

------------------------------------------------------------------------

## 📂 Project Structure

    agentic-rag-student-reasoning/
    ├── agent/                  # Core agent implementations
    │   ├── orchestrator.py
    │   ├── extractor.py
    │   ├── classifier.py
    │   ├── aggregator.py
    │   └── reporter.py
    ├── api/                    # FastAPI endpoints
    │   └── main.py
    ├── cli/                    # Command-line tools
    │   └── cli.py
    ├── sql/                    # Database schema + migrations
    │   └── schema.sql
    ├── data/                   # Example CSVs (questions, rubric, student answers)
    ├── tests/                  # Unit + integration tests
    ├── planning.md             # Project plan
    └── README.md               # Project documentation


------------------------------------------------------------------------

## 🚀 Quick Start

### 1. Setup Environment

``` bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup PostgreSQL

``` bash
# Create database
createdb agentic_rag

# Run schema
psql -d agentic_rag -f sql/schema.sql
```

> ⚠️ Adjust embedding dimensions in `schema.sql` based on your embedding
> model (e.g., 384, 768 or 1536).

### 3. Configure `.env`

``` env
DATABASE_URL=postgresql://username:password@localhost:5432/agentic_rag

LLM_PROVIDER=openai
LLM_API_KEY=sk-your-key
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

APP_ENV=development
LOG_LEVEL=INFO
```

### 4. Ingest Data

``` bash
# Import professor-defined rubric, criteria, keywords, and questions
python cli/cli.py ingest --file data/professor_inputs.csv

# Import student answers
python cli/cli.py ingest --file data/student_answers.csv
```

### 5. Run Batch Processing

``` bash
python cli/cli.py run --question 1 --batch-size 10 --threshold 0.8
```

------------------------------------------------------------------------

## 🔄 Workflow

1.  **Ingestion**: Professors provide questions, rubric, criteria, keywords → stored in DB.
2.  **Batch Processing**: Orchestrator splits student answers into batches.
3.  **Agents**:
    -   Extracting Agent → themes/keywords
    -   Classifier Agent → Standard / Latent / Off-topic
    -   Reporter → reasoning + summary
    -   Aggregator → novel patterns
4.  **Professor Review**: Proposed new keywords/themes require approval.
5.  **Outputs**: JSON/CSV reports + aggregated insights.

------------------------------------------------------------------------

## 📊 Evaluation

-   Gold set of labeled answers maintained for benchmarking
-   Metrics:
    -   Precision/Recall/F1 per class
    -   Retry rate
    -   Confidence distribution
    -   Keyword drift rate
-   Low-confidence cases prioritized for human review

------------------------------------------------------------------------

## 🧪 Testing

``` bash
pytest
pytest --cov=agent --cov=api --cov-report=html
```

------------------------------------------------------------------------

## 🔧 Troubleshooting

-   **No results?** Ensure `ingest` step has been run.
-   **DB errors?** Confirm `DATABASE_URL` in `.env`.
-   **Low performance?** Check embedding dimensions in schema + index setup.

------------------------------------------------------------------------
Built with ❤️ using **FastAPI**, **PostgreSQL + pgvector**, and **multi-agent orchestration**.
