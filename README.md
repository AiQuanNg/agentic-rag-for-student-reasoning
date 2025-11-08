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

```
agentic-rag-student-reasoning/
├── agent/                  # Core agent implementations
│   ├── __init__.py
│   ├── orchestrator.py     # ✅ Python-based workflow manager
│   ├── extractor.py        # 🆕 Pydantic AI agent (NEW)
│   ├── classifier.py       # ⏳ Build after Extractor
│   ├── reporter.py         # ✅ Summary Reporter
│   ├── aggregator.py       # ⏳ Build after Classifier
│   │
│   ├── tools/              # 🆕 Tools Layer (NEW)
│   │   ├── __init__.py
│   │   ├── codebook.py     # Keyword retrieval tools
│   │   └── retrieval.py    # pgvector search utilities
│   │
│   ├── prompts/            # 🆕 Prompt Engineering (NEW)
│   │   ├── __init__.py
│   │   ├── extractor.py    # EXTRACTOR_SYSTEM_PROMPT
│   │   ├── classifier.py   # (Future)
│   │   └── aggregator.py   # (Future)
│   │
│   └── models/             # 🆕 Validation Layer (NEW)
│       ├── __init__.py
│       ├── extraction.py   # ExtractionResult, ExtractorContext
│       ├── classification.py    # (Future)
│       └── common.py            # Shared models
│
├── config/                  # Configuration files
│   └── extractor_config.py  # ✅ Extractor configuration
│
├── scripts/                         # Utility scripts
│   └── generate_embeddings.py       # ✅ Generate embeddings for DB
│
├── data/                        # Data files
│   ├── questions.csv            # ✅ Questions dataset
│   ├── rubrics.csv              # ✅ Rubrics dataset
│   ├── criteria.csv             # ✅ Criteria dataset
│   ├── topic_keywords.csv       # ✅ Keywords dataset
│   └── student_answers.csv      # ✅ Student submissions
│
├── sql/                      # Database schemas
│   ├── schema.sql            # ✅ PostgreSQL + pgvector
│   ├── constraints.sql       # ✅ Database constraints
│   └── clean.sql             # ✅ Cleanup scripts
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_extractor.py            # 🆕 Unit tests
│   ├── test_codebook_tools.py       # 🆕 Tool tests
│   └── test_extraction_models.py    # 🆕 Validation tests
│
├── planning.md           # � Project planning document
├── TASK.md               # � Task tracking
├── README.md             # � Project documentation
├── requirements.txt      # � Python dependencies
└── pytest.ini            # ⚙️ Pytest configuration
```



## 🏗️ Architecture (Following Ottomator Pattern)

This system adopts a **layered agentic architecture** inspired by production RAG systems:

### Five-Layer Design

#### 1. **User Layer**
- **FastAPI**: Orchestrator endpoints (`POST /runs`, `GET /runs/{id}/stream`)
- **CLI**: Development testing tools

#### 2. **Agent Layer** (LLM Reasoning)
**Pydantic AI Agents**: Type-safe, tool-enabled LLM agents
- **Extractor**: GPT-4o-mini with codebook retrieval tools
- **Classifier**: GPT-4o with rubric retrieval tools (future)
- **Aggregator**: Pattern detection agent (future)

#### 3. **Tools Layer** (Retrieval Functions)
**Codebook Tools**: Keyword retrieval from approved thematic codebook
- `retrieve_codebook_keywords(question_id)` - String-based lookup
- `search_similar_keywords(embedding)` - Optional pgvector search

**Rubric Tools** (future):
- `retrieve_rubric(answer_embedding, top_k)` - pgvector similarity

**Criteria Tools** (future):
- `retrieve_criteria(answer_embedding, top_k)` - pgvector similarity

#### 4. **Database Layer**
**PostgreSQL + pgvector**: Vector similarity search
- `topic_keywords.embedding` - 384-dim (all-MiniLM-L6-v2)
- `rubrics.embedding`, `criteria.embedding`
- **asyncpg**: Async database connection pool

#### 5. **Validation Layer**
**Pydantic Models**: Structured input/output validation
- `ExtractionResult`: Enforces schema compliance for Extractor outputs
- `ClassificationResult`: (future)

### Agentic Loop Pattern

```
User → Orchestrator → Extractor Agent
                          ↓
              [Tools: retrieve_codebook_keywords]
                          ↓
              PostgreSQL + pgvector
                          ↓
              Structured ExtractionResult
                          ↓
              Classifier Agent (future)
```

**Key Insight**: Unlike LLM-based orchestration, our **Orchestrator uses Python logic** for workflow control (deterministic), while **agents use LLMs** for semantic reasoning (probabilistic).

### Extractor Agent Architecture

```python
# agent/extractor.py
extractor_agent = Agent(
    model=OpenAIModel("gpt-4o-mini"),      # Cost-efficient for extraction
    system_prompt=EXTRACTOR_SYSTEM_PROMPT, # 280+ lines with tool guidance
    result_type=ExtractionResult           # Pydantic validation
)

@extractor_agent.tool
async def retrieve_codebook_keywords(ctx: RunContext) -> List[str]:
    """LLM calls this to get approved keywords"""
    return await db.fetch_keywords(ctx.deps.question_id)

# LLM decides when/how to use tools based on system prompt
result = await extractor_agent.run(
    f"Extract keywords from: {student_answer}",
    deps=ExtractorContext(db_pool=pool, question_id=1)
)
```

### Why Two Models?

| **Component** | **Model** | **Purpose** |
|---------------|-----------|-------------|
| **Embedding** | Sentence-Transformers<br/>(`all-MiniLM-L6-v2`) | Convert text → vectors for pgvector similarity |
| **Reasoning** | GPT-4o-mini/GPT-4o<br/>(via Pydantic AI) | Semantic understanding, keyword matching, classification |

**Embeddings are pre-computed** (stored in DB), **LLMs do reasoning** at runtime.

### Classifier Agent Architecture (Hybrid Approach)

**Challenge**: Generic rubrics are not semantically rich enough for pure pgvector retrieval.

**Solution**: Use a hybrid classifier that combines the Extractor's structured output with LLM reasoning and rubric context.

```python
# agent/classifier.py (illustrative example)
from typing import Dict
from pydantic_ai import Agent

# Replace these imports with the project's actual modules
# from agent.models.classification import ClassificationResult
# from agent.contexts import RunContext, ClassifierContext

classifier_agent = Agent(
    model=OpenAIModel("gpt-4o"),
    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    result_type=ClassificationResult,
)

@classifier_agent.tool
async def get_question_rubrics(ctx: RunContext) -> Dict:
    """Fetch all rubric levels (no similarity filtering)."""
    return await db.fetch_all_rubrics(ctx.deps.question_id)

@classifier_agent.tool
async def get_classification_criteria(ctx: RunContext) -> Dict:
    """Fetch human-readable Standard vs Latent criteria definitions."""
    return await db.fetch_criteria()
```

Classifier uses the Extractor's structured output (keywords, themes, novel terms, confidence) as a decision proxy. Example usage:

```python
# example usage: run the classifier with extractor context
result = await classifier_agent.run(
    f"""Classify answer:
ANSWER: {student_answer}

EXTRACTOR FOUND:
- Keywords: {extraction.matched_keywords}
- Themes: {extraction.detected_themes}
- Novel Terms: {extraction.novel_terms}
- Confidence: {extraction.extraction_confidence}

Determine if STANDARD, LATENT, or OFF_TOPIC.
""",
    deps=ClassifierContext(db_pool=pool, question_id=1),
)
```

**Why Hybrid?**

| Aspect | Pure pgvector | Hybrid Approach |
|---|---:|:---|
| Rubric matching | Low/ambiguous similarity | LLM reads rubric descriptors and judges fit |
| Standard detection | Misses semantic signal | Uses keyword-count proxy and rubric alignment |
| Latent detection | Often conflated with Standard | Uses `novel_terms` + `detected_themes` signals |
| Interpretability | "Why 0.52 similarity?" | Clear decision logic and reasoning traces |

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
EMBEDDING_MODEL=all-MiniLM-L6-v2

APP_ENV=development
LOG_LEVEL=INFO
```

### 4. Ingest Data

``` bash
# Import professor-defined rubric, criteria, keywords, and questions

# Import student answers

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
    -   Classifier Agent → Standard / Latent / Off-topic + reasoning
    -   Reporter → summary
    -   Aggregator → novel patterns
4.  **Professor Review**: Review new keywords/themes require approval.
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

-   **DB errors?** Confirm `DATABASE_URL` in `.env`.
-   **Low performance?** Check embedding dimensions in schema + index setup.

------------------------------------------------------------------------
Built with ❤️ using **FastAPI**, **PostgreSQL + pgvector**, and **multi-agent orchestration**.
