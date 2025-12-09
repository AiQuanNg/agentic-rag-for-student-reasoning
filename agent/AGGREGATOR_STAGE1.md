# Aggregator Stage 1: Term Clustering

## Overview

Stage 1 performs bottom-up term clustering from classified student answers. It groups semantically similar novel terms to identify curriculum gaps and emerging vocabulary patterns.

## Configuration

Add these variables to your `.env` file:

```bash
# Aggregator Stage 1: Term Extractor
# Uses simple, fast model for term equivalence validation
AGGREGATOR_STAGE1_PROVIDER=openrouter
AGGREGATOR_STAGE1_API_KEY=your-api-key-here
AGGREGATOR_STAGE1_MODEL=openai/gpt-4o-mini
AGGREGATOR_STAGE1_TEMPERATURE=0.0
```

### Model Recommendations

**Fast & Cheap (Recommended):**
- `openai/gpt-4o-mini` - Good for term validation
- `google/gemini-flash-1.5` - Fast and accurate

**High Quality (If needed):**
- `deepseek/deepseek-chat` - Better semantic understanding
- `openai/gpt-4-turbo` - Most accurate but expensive

## Usage

### Run Test on Golden Dataset

```bash
python -m cli.test_aggregator_stage1
```

This will:
1. Load the golden dataset from `tests/classifer_test_batches/`
2. Extract all novel terms from routed answers
3. Group semantically similar terms using LLM validation
4. Generate term clusters with recommendations
5. Export results to `tests/stage1_term_clusters.csv`

### Programmatic Usage

```python
import asyncio
import pandas as pd
from agent.aggregator_stage1 import TermExtractor

async def run_stage1():
    # Load classified results
    df = pd.read_csv('path/to/classified_results.csv')
    
    # Run Stage 1
    extractor = TermExtractor()
    result = await extractor.process(df)
    
    # Access clusters
    for cluster in result.term_clusters:
        print(f"{cluster.canonical_term}: {cluster.frequency} occurrences")
        print(f"  Recommendation: {cluster.recommendation}")

asyncio.run(run_stage1())
```

## Input Requirements

The input DataFrame must contain these columns from the classifier:

| Column | Description |
|--------|-------------|
| `aggregator_recommendation` | Filter for 'ROUTE' to process |
| `high_priority_terms` | Pipe-separated high-priority novel terms |
| `medium_priority_terms` | Pipe-separated medium-priority novel terms |
| `answer_id` | Unique answer identifier |
| `answer_text` | Full student answer text (for evidence extraction) |
| `topic` | Topic context (optional) |

## Output Format

### Stage1Output Structure

```python
{
    "term_clusters": [
        {
            "cluster_id": "TC001",
            "canonical_term": "attention mechanisms",
            "variants": ["attention mechanisms", "attention mechanism"],
            "frequency": 4,
            "unique_students": 4,
            "student_ids": ["1293", "1302", "1304", "1309"],
            "topics": ["How generative AI gathers information"],
            "evidence_quotes": [
                {
                    "answer_id": "1293",
                    "term": "attention mechanisms",
                    "quote": "...The transformer architecture uses attention mechanisms..."
                }
            ],
            "recommendation": "ADD_TO_KB",
            "recommendation_reason": "4 students using similar terms - strong pattern",
            "priority_breakdown": {"high": 4, "medium": 0}
        }
    ],
    "statistics": {
        "total_clusters": 10,
        "add_to_kb_count": 2,
        "review_count": 3,
        "monitor_count": 5,
        "total_unique_students": 17,
        "high_priority_terms": 45,
        "medium_priority_terms": 28
    },
    "total_terms_processed": 73,
    "total_answers_processed": 17
}
```

### CSV Export Columns

The test CLI exports these columns to `tests/stage1_term_clusters.csv`:

- `cluster_id` - Unique identifier (TC001, TC002, ...)
- `canonical_term` - Most representative term
- `variants` - All term variations (pipe-separated)
- `frequency` - Total occurrences
- `unique_students` - Number of unique students
- `student_ids` - Student answer IDs (pipe-separated)
- `topics` - Topic contexts (pipe-separated)
- `high_priority_count` - High-priority occurrences
- `medium_priority_count` - Medium-priority occurrences
- `recommendation` - Action: ADD_TO_KB | REVIEW | MONITOR
- `recommendation_reason` - Explanation
- `evidence_quote_1` - Sample quote 1
- `evidence_quote_2` - Sample quote 2

## Recommendation Thresholds

| Unique Students | Recommendation | Meaning |
|-----------------|----------------|---------|
| ≥ 5 | ADD_TO_KB | Strong pattern - add to knowledge base |
| 3-4 | REVIEW | Emerging pattern - needs professor review |
| < 3 | MONITOR | Low frequency - monitor for growth |

## How It Works

### 1. Term Extraction
- Reads `high_priority_terms` and `medium_priority_terms` columns
- Splits on pipe delimiter (`|`)
- Creates inventory of all term occurrences

### 2. Variant Normalization
- Groups identical terms (case-insensitive)
- Uses sentence embeddings to find similar terms (cosine similarity > 0.85)
- Validates candidates with LLM: "Are these terms equivalent?"
- Merges validated variants into canonical clusters

### 3. Cluster Creation
- Groups by canonical term
- Counts total frequency and unique students
- Filters out single occurrences (frequency < 2)
- Assigns recommendations based on thresholds

### 4. Evidence Enrichment
- Samples 2 students per cluster
- Extracts quotes containing the term
- Adds context for professor review

## LLM Validation Logic

The LLM validator uses this system prompt:

```
You are a linguistic expert specializing in AI/ML terminology.
Determine if two terms are semantically equivalent in technical contexts.

Consider:
- Synonyms (e.g., "neural net" = "neural network")
- Technical vs colloquial (e.g., "token" = "word")
- Abbreviations (e.g., "AI" = "artificial intelligence")
- Hyphenation variants (e.g., "token-by-token" = "token by token")

DO NOT consider equivalent:
- Different concepts (e.g., "training" ≠ "inference")
- Different specificity (e.g., "neural network" ≠ "transformer")
```

## Performance

On the golden dataset (32 answers, ~17 routed):
- Processing time: ~30-60 seconds
- LLM calls: ~20-40 validation calls
- Cost: ~$0.01-0.03 (using GPT-4o-mini)

## Troubleshooting

### No clusters found
- Check if answers are being routed (`aggregator_recommendation == 'ROUTE'`)
- Check if classifier is extracting novel terms
- All terms might be unique (frequency < 2)

### Too many clusters
- Lower the embedding similarity threshold (currently 0.85)
- Use a stricter LLM model
- Increase frequency threshold (currently 2)

### LLM validation errors
- Check API key configuration
- Verify model availability on your provider
- Check internet connection
- Review logs for specific error messages

## Next Steps

After Stage 1, the clusters can be:
1. Exported for professor review
2. Passed to Stage 3 (Curator) for final formatting
3. Integrated with Stage 2 (Theme Synthesizer) outputs
