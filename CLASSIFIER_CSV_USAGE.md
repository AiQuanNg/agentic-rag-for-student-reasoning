# Using Pre-computed Extractor Results with Classifier

## Overview

The `test_classifier.py` script now supports loading pre-computed extractor results from CSV files. This allows you to:

1. **Save time**: Skip the extraction step when you already have results
2. **Save API costs**: Don't re-run the extractor LLM calls
3. **Test classifier variations**: Try different classifier prompts/models on the same extraction data
4. **Reproducibility**: Ensure consistent extraction results across multiple classifier tests

## Usage

### Basic Command (New Feature)

```bash
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --limit 32
```

### Options

- `--question-id`: Question ID (1-4) - **REQUIRED**
- `--extractor-csv`: Path to extractor results CSV - **NEW OPTION**
- `--limit`: Number of answers to process (default: 10)
- `--students`: Specific student IDs (e.g., `S1001-S1010` or `S1001,S1002,S1003`)
- `--output`: Custom output CSV path (optional)
- `--db-url`: Database URL (optional, uses DATABASE_URL env var)

## How It Works

### With Extractor CSV (Fast Mode)

```bash
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --limit 32
```

**Pipeline:**
1. ✅ Load extractor results from CSV (instant)
2. ✅ Fetch question context from database
3. ✅ Run classifier agent (LLM call)
4. ✅ Save combined results to CSV

**Advantage:** Skip extractor LLM calls = **~50% faster + 50% cheaper**

### Without Extractor CSV (Full Pipeline Mode)

```bash
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --limit 10
```

**Pipeline:**
1. ✅ Fetch answers from database
2. ✅ Run extractor agent (LLM call)
3. ✅ Run classifier agent (LLM call)
4. ✅ Save combined results to CSV

## Expected CSV Format

The extractor CSV must contain these columns:

### Required Columns:
- `answer_id`: Integer ID
- `question_id`: Integer ID
- `extraction_confidence`: Float (0.0-1.0)

### Extraction Result Columns (pipe-separated lists):
- `matched_keywords`: e.g., `ai|machine learning|neural networks`
- `detected_themes`: e.g., `technical|ethical`
- `novel_terms`: e.g., `gradient descent|backpropagation`
- `topic`: e.g., `AI definition|How AI works`
- `evidence_spans`: e.g., `quote 1|quote 2|quote 3`

### Example CSV Row:

```csv
answer_id,question_id,extraction_confidence,matched_keywords,detected_themes,novel_terms,topic,evidence_spans
1281,1,0.85,ai|bias|regulation,ethical|business,job displacement|transparency requirements,AI ethics|Societal impact,people are worried about AI|companies should be allowed to use AI
```

## Use Cases

### 1. Test Different Classifier Models

```bash
# Run extractor once
.venv/bin/python cli/batch_test_extractor.py --question-id 1 --total-students 32 --batch-size 5

# Test with different classifier models (reuse extractor results)
# Model 1: nvidia/nemotron-nano
.venv/bin/python cli/test_classifier.py --question-id 1 --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv

# Switch to Model 2 in agent/config/providers.py
# Model 2: deepseek/deepseek-v3.2
.venv/bin/python cli/test_classifier.py --question-id 1 --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv
```

### 2. Test Classifier Prompt Changes

```bash
# Edit agent/prompts/classifier.py (update system prompt)

# Test new prompt without re-running extractor
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --output tests/classifier_new_prompt.csv
```

### 3. Batch Processing with Cached Extraction

For your current 32 students (S1001-S1032):

```bash
# You already have: extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv

# Run classifier on all 32 students using cached extraction
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --limit 32 \
  --output tests/classifier_Q1_S1001-S1032_cached.csv
```

## Performance Comparison

### Full Pipeline (Extractor + Classifier)
- **Time**: ~30-60 seconds for 5 answers
- **API Calls**: 10 LLM calls (5 extractor + 5 classifier)
- **Cost**: 2× API costs

### With Cached Extractor Results
- **Time**: ~15-30 seconds for 5 answers
- **API Calls**: 5 LLM calls (only classifier)
- **Cost**: 1× API costs (50% savings)

## Output

The output CSV will include **BOTH** extractor and classifier results:

### Columns (33 total):
1. **Metadata** (4): answer_id, question_id, status, answer_text
2. **Extraction** (9): extraction_confidence, topic_count, topic, matched_keywords_count, detected_themes_count, novel_terms_count, matched_keywords, detected_themes, novel_terms
3. **Classification** (5): classification_label, classification_confidence, reasoning, evidence_spans_count, evidence_spans
4. **Rubric Assessment** (4): rubric_level_achieved, rubric_level_100, rubric_level_50, rubric_level_0
5. **Novel Term Flagging** (3): flagged_novel_terms_count, high_priority_terms, medium_priority_terms
6. **Latent Signals** (3): latent_mechanism_explanations, latent_novel_terms_in_mechanisms, latent_critical_engagement
7. **Aggregator Routing** (2): aggregator_recommendation, aggregator_reason
8. **Error Tracking** (1): error

## Troubleshooting

### Issue: "answer_id not found in extractor cache"

**Problem:** The CSV doesn't contain the answer IDs you're testing

**Solution:** Make sure the extractor CSV covers all the student answers you want to test
```bash
# Check which answer_ids are in the CSV
cut -d',' -f1 tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv | tail -n +2
```

### Issue: "Failed to load extractor results from CSV"

**Problem:** CSV format doesn't match expected structure

**Solution:** Ensure your CSV has these columns:
- answer_id
- question_id
- extraction_confidence
- matched_keywords
- detected_themes
- novel_terms
- topic
- evidence_spans

### Issue: Mixed mode (some answers cached, some not)

**Behavior:** If an answer_id is NOT in the CSV, the script will run the extractor agent for that answer

**Solution:** This is intentional! You can partially cache results and fill in missing ones.

## Example Workflow

```bash
# Step 1: Run extractor on all students (one-time, save results)
.venv/bin/python cli/batch_test_extractor.py \
  --question-id 1 \
  --start-student 1001 \
  --total-students 32 \
  --batch-size 5

# Output: tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv

# Step 2: Test classifier using cached extractor results (fast, reusable)
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --limit 32 \
  --output tests/classifier_test1.csv

# Step 3: Modify classifier prompt in agent/prompts/classifier.py

# Step 4: Re-test classifier with SAME extractor results (compare outputs)
.venv/bin/python cli/test_classifier.py \
  --question-id 1 \
  --extractor-csv tests/extractor_Q1_S1001-S1032_20251203_134445_deepseekv32.csv \
  --limit 32 \
  --output tests/classifier_test2.csv

# Step 5: Compare results
# tests/classifier_test1.csv vs tests/classifier_test2.csv
# Both use SAME extraction, different classification
```

## Benefits

1. **Reproducibility**: Same extraction results = consistent comparison
2. **Speed**: Skip expensive LLM calls for extraction
3. **Cost**: 50% API cost savings
4. **Flexibility**: Test multiple classifier configurations quickly
5. **Debugging**: Isolate classifier behavior from extractor variations

## Notes

- The extractor CSV is **read-only** - original file is never modified
- If answer_id is not in CSV, script falls back to running extractor agent
- Compatible with both `test_classifier.py` and future batch versions
- CSV loading happens once at startup (cached in memory)
