"""
Test CLI for Aggregator Stage 1: Term Extractor

Tests term clustering on classifier output and displays results.

Usage:
    python -m cli.test_aggregator_stage1 [input_csv] [--output-dir OUTPUT_DIR]

Arguments:
    input_csv       Path to classifier output CSV file (optional)
                    Default: tests/classifer_test_batches/merged_results_Q1_S1001-S1032_20251205_162823_deepseekv32.csv
    --output-dir    Directory to save output files (optional)
                    Default: tests/aggregator_stage1_output/

Examples:
    # Use default paths
    python -m cli.test_aggregator_stage1
    
    # Specify input CSV
    python -m cli.test_aggregator_stage1 data/classifier_results.csv
    
    # Specify both input and output directory
    python -m cli.test_aggregator_stage1 data/classifier_results.csv --output-dir results/stage1/

Output:
    - Console summary of term clusters
    - CSV exports to specified output directory:
        * stage1_term_clusters_TIMESTAMP.csv
        * stage1_statistics_TIMESTAMP.csv
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
import pandas as pd
import argparse
from datetime import datetime

from agent.aggregator_stage1 import TermExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_stage1(input_csv: Path, output_dir: Path):
    """
    Test Stage 1 term extractor on classifier output.
    
    Args:
        input_csv: Path to classifier output CSV file
        output_dir: Directory to save output files
    """
    
    print("\n" + "="*70)
    print("AGGREGATOR STAGE 1: TERM CLUSTERING TEST")
    print("="*70 + "\n")
    
    # Validate input file
    if not input_csv.exists():
        print(f"❌ Error: Input CSV not found at {input_csv}")
        return
    
    print(f"📂 Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"   Total answers in dataset: {len(df)}")
    
    # Check for required columns
    required_cols = ['aggregator_recommendation', 'high_priority_terms', 'medium_priority_terms', 'answer_id', 'answer_text']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Error: Missing required columns: {missing}")
        return
    
    # Count routed answers
    routed_count = len(df[df['aggregator_recommendation'] == 'ROUTE'])
    print(f"   Routed to Aggregator: {routed_count} answers\n")
    
    if routed_count == 0:
        print("⚠️  Warning: No answers routed to aggregator!")
        print("   Check if classifier is flagging answers with 'ROUTE' recommendation.\n")
        return
    
    # Initialize and run Stage 1
    print("🔄 Processing Stage 1: Term Clustering...")
    print("   (This may take a few minutes for LLM validation)\n")
    
    extractor = TermExtractor()
    
    try:
        result = await extractor.process(df)
        
        # Display summary statistics
        print("\n" + "="*70)
        print("📊 SUMMARY STATISTICS")
        print("="*70 + "\n")
        
        stats = result.statistics
        print(f"Answers Processed:     {result.total_answers_processed}")
        print(f"Terms Extracted:       {result.total_terms_processed}")
        print(f"  ├─ High Priority:    {stats.high_priority_terms}")
        print(f"  └─ Medium Priority:  {stats.medium_priority_terms}")
        print(f"\nTerm Clusters:         {stats.total_clusters}")
        print(f"  ├─ ADD_TO_KB:        {stats.add_to_kb_count}")
        print(f"  ├─ REVIEW:           {stats.review_count}")
        print(f"  └─ MONITOR:          {stats.monitor_count}")
        print(f"\nUnique Students:       {stats.total_unique_students}")
        
        # Display processing metadata
        print(f"\nProcessing Time:       {result.processing_metadata.get('processing_time_seconds', 0):.2f}s")
        print(f"Model Used:            {result.processing_metadata.get('model_used', 'N/A')}")
        
        # Display term clusters
        print("\n" + "="*70)
        print("🔍 TERM CLUSTERS DETAIL")
        print("="*70 + "\n")
        
        if not result.term_clusters:
            print("⚠️  No term clusters found!")
            print("   This could mean:")
            print("   - All terms were unique (frequency < 2)")
            print("   - LLM didn't find any semantic equivalences")
            print("   - Not enough routed answers\n")
        else:
            for cluster in result.term_clusters:
                print(f"\n{cluster.cluster_id}: {cluster.canonical_term}")
                print(f"{'─'*70}")
                print(f"Recommendation:  {cluster.recommendation}")
                print(f"Reason:          {cluster.recommendation_reason}")
                print(f"Frequency:       {cluster.frequency} occurrences")
                print(f"Unique Students: {cluster.unique_students}")
                print(f"Variants:        {', '.join(cluster.variants)}")
                print(f"Priority:        High={cluster.priority_breakdown.get('high', 0)}, Medium={cluster.priority_breakdown.get('medium', 0)}")
                
                if cluster.topics:
                    print(f"Topics:          {', '.join(cluster.topics[:3])}")  # Show first 3
                
                if cluster.evidence_quotes:
                    print(f"\nEvidence Quotes:")
                    for i, quote_data in enumerate(cluster.evidence_quotes[:2], 1):
                        print(f"  {i}. [Answer {quote_data['answer_id']}]")
                        print(f"     {quote_data['quote'][:150]}...")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to CSV
        output_file = output_dir / f'stage1_term_clusters_{timestamp}.csv'
        
        print("\n" + "="*70)
        print("💾 EXPORTING RESULTS")
        print("="*70 + "\n")
        
        # Prepare data for CSV export
        export_data = []
        for cluster in result.term_clusters:
            export_data.append({
                'cluster_id': cluster.cluster_id,
                'canonical_term': cluster.canonical_term,
                'variants': '|'.join(cluster.variants),
                'frequency': cluster.frequency,
                'unique_students': cluster.unique_students,
                'student_ids': '|'.join(cluster.student_ids),
                'topics': '|'.join(cluster.topics) if cluster.topics else '',
                'high_priority_count': cluster.priority_breakdown.get('high', 0),
                'medium_priority_count': cluster.priority_breakdown.get('medium', 0),
                'recommendation': cluster.recommendation,
                'recommendation_reason': cluster.recommendation_reason,
                'evidence_quote_1': cluster.evidence_quotes[0]['quote'] if len(cluster.evidence_quotes) > 0 else '',
                'evidence_quote_2': cluster.evidence_quotes[1]['quote'] if len(cluster.evidence_quotes) > 1 else ''
            })
        
        if export_data:
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(output_file, index=False)
            print(f"✅ Results exported to: {output_file}")
            print(f"   {len(export_data)} term clusters saved")
        else:
            print("⚠️  No clusters to export")
        
        # Export summary stats
        stats_file = output_dir / f'stage1_statistics_{timestamp}.csv'
        stats_df = pd.DataFrame([{
            'timestamp': result.processing_metadata.get('timestamp', ''),
            'total_answers_processed': result.total_answers_processed,
            'total_terms_processed': result.total_terms_processed,
            'total_clusters': stats.total_clusters,
            'add_to_kb_count': stats.add_to_kb_count,
            'review_count': stats.review_count,
            'monitor_count': stats.monitor_count,
            'unique_students': stats.total_unique_students,
            'high_priority_terms': stats.high_priority_terms,
            'medium_priority_terms': stats.medium_priority_terms,
            'processing_time_seconds': result.processing_metadata.get('processing_time_seconds', 0),
            'model_used': result.processing_metadata.get('model_used', '')
        }])
        stats_df.to_csv(stats_file, index=False)
        print(f"✅ Statistics exported to: {stats_file}\n")
        
        print("="*70)
        print("✅ STAGE 1 TEST COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during Stage 1 processing:")
        print(f"   {type(e).__name__}: {e}")
        logger.exception("Stage 1 processing failed")
        raise


def main():
    """Main entry point."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Test Aggregator Stage 1: Term Clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths
  python -m cli.test_aggregator_stage1
  
  # Specify input CSV
  python -m cli.test_aggregator_stage1 data/classifier_results.csv
  
  # Specify both input and output directory
  python -m cli.test_aggregator_stage1 data/classifier_results.csv --output-dir results/stage1/
        """
    )
    
    parser.add_argument(
        'input_csv',
        nargs='?',
        type=str,
        default='tests/classifer_test_batches/merged_results_Q1_S1001-S1032_20251205_162823_deepseekv32.csv',
        help='Path to classifier output CSV file (default: tests/classifer_test_batches/merged_results_Q1_S1001-S1032_20251205_162823_deepseekv32.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='tests/aggregator_stage1_output',
        help='Directory to save output files (default: tests/aggregator_stage1_output/)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = project_root / input_csv
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    try:
        asyncio.run(test_stage1(input_csv, output_dir))
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
