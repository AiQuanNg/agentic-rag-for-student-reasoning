# Batch Test Classifier - Usage Guide

## Overview

The `batch_test_classifier.py` script now properly handles batch testing of the classifier agent with automatic merging of results into a single CSV file.

## What Was Fixed

### Previous Issues:
1. The `test_batch` method was returning results list instead of saving to a file
2. Individual batch files weren't being created with proper naming
3. Merge functionality existed but wasn't receiving the batch file paths

### Updates Made:
1. ✅ Modified `run_batch()` to properly save batch results to individual CSV files
2. ✅ Each batch now creates a file named: `batch_results_Q{question_id}_{start_student_id}_{timestamp}.csv`
3. ✅ All batch files are automatically merged into one consolidated file at the end
4. ✅ Merged file is named: `merged_results_Q{question_id}_S0001-S{total_students}_{timestamp}.csv`

## Usage Examples

### Test 10 students in batches of 5:
```bash
python cli/batch_test_classifier.py --question-id 1 --total-students 10 --batch-size 5
```

This will:
- Process students S0001-S0005 (Batch 1) → saves to `batch_results_Q1_S0001_TIMESTAMP.csv`
- Wait 2 seconds (default delay)
- Process students S0006-S0010 (Batch 2) → saves to `batch_results_Q1_S0006_TIMESTAMP.csv`
- Merge both files into → `merged_results_Q1_S0001-S0010_TIMESTAMP.csv`

### Test 50 students in batches of 10:
```bash
python cli/batch_test_classifier.py --question-id 1 --total-students 50 --batch-size 10
```

This creates 5 batches (10 students each) and merges them into one file.

### Custom delay between batches:
```bash
python cli/batch_test_classifier.py --question-id 1 --total-students 20 --batch-size 5 --delay 5
```

This adds a 5-second delay between batches (useful for rate limiting).

## Output Files

### Individual Batch Files
Located in `tests/` directory:
- `batch_results_Q1_S0001_20251105_145904.csv` (Batch 1: S0001-S0005)
- `batch_results_Q1_S0006_20251105_145920.csv` (Batch 2: S0006-S0010)
- etc.

These contain the full classification results for each batch.

### Merged File
Located in `tests/` directory:
- `merged_results_Q1_S0001-S0050_20251105_150500.csv`

This is the **consolidated file** containing all results from all batches, sorted by `answer_id`.

## Output Summary

The script provides:
1. **Real-time progress** with progress bars
2. **Batch completion status** (successful/failed batches)
3. **Merged file statistics**:
   - Total records processed
   - Classification distribution (STANDARD, LATENT, OFF_TOPIC)
   - Average confidence scores
4. **File locations** for both individual batches and merged results

## Example Output

```
Starting Batch Classification Test
Question ID: 1
Total Students: 10 (S0001 - S0010)
Batch Size: 5
Number of Batches: 2
Delay Between Batches: 2s

Processing 2 batches... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00

===== BATCH TEST COMPLETE =====
Total Batches: 2
Successful: 2
Failed: 0
Success Rate: 100.0%

Merging 2 batch result files...
  • Loaded tests/batch_results_Q1_S0001_20251105_145904.csv: 5 rows
  • Loaded tests/batch_results_Q1_S0006_20251105_145920.csv: 5 rows

Merged Results Summary
Total Records: 10

Classification Distribution:
  • STANDARD: 6 (60.0%)
  • LATENT: 3 (30.0%)
  • OFF_TOPIC: 1 (10.0%)

Average Confidence: 0.85

✓ All results merged into: tests/merged_results_Q1_S0001-S0010_20251105_150500.csv

Individual batch results: tests/batch_results_Q1_S*.csv
```

## CSV Structure

Both individual batch files and the merged file contain the same columns:

| Column | Description |
|--------|-------------|
| `answer_id` | Unique student submission ID |
| `question_id` | Question number (1-4) |
| `status` | Processing status (success/error) |
| `classification_label` | Primary classification (standard/latent/off_topic) |
| `classification_confidence` | Confidence score (0.0-1.0) |
| `reasoning` | Explanation for the classification |
| `matched_keywords` | Keywords found in the answer |
| `detected_themes` | Themes identified by extractor |
| `novel_terms` | Novel terminology detected |
| `evidence_spans` | Text spans supporting classification |
| `aggregator_recommendation` | Whether to route to aggregator (ROUTE/BASELINE) |
| ... | Additional extraction and rubric metadata |

## Notes

- The script automatically handles database connections and disconnections
- Each batch is processed independently to avoid rate limits
- Failed batches don't prevent other batches from completing
- The merged file is sorted by `answer_id` for consistency
- Individual batch files are preserved for debugging/analysis
