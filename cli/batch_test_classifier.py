"""
Batch Test Classifier - Run classifier on multiple batches of students.

This script runs the classifier test in batches to avoid rate limits
and provide progress updates.

Usage:
    python cli/batch_test_classifier.py --question-id 1 --total-students 50 --batch-size 5
"""

import asyncio
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# Import ClassifierTester from test_classifier
from cli.test_classifier import ClassifierTester

load_dotenv()
console = Console()


async def run_batch(question_id: int, start_student_id: str, batch_size: int) -> str:
    """
    Run a single batch of student classification tests.
    
    Args:
        question_id: Question ID to test
        start_student_id: Starting student ID (e.g., 'S0001')
        batch_size: Number of students in this batch
    
    Returns:
        Path to output CSV file if successful, None otherwise
    """
    try:
        # Get database URL from environment
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            console.print("[red]DATABASE_URL not found in environment[/red]")
            return None
        
        # Create tester instance and connect to DB
        tester = ClassifierTester(db_url=db_url)
        await tester.connect_db()
        
        # Generate list of student IDs for this batch
        start_num = int(start_student_id[1:])  # Extract number from 'S0001'
        student_ids = [f"S{i:04d}" for i in range(start_num, start_num + batch_size)]
        
        # Run the batch test
        results = await tester.test_batch(
            question_id=question_id,
            student_ids=student_ids
        )
        
        # Save results to batch-specific file
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"tests/batch_results_Q{question_id}_{start_student_id}_{timestamp}.csv"
            output_path = tester.save_results(output_path)
        else:
            output_path = None
        
        # Disconnect from DB
        await tester.disconnect_db()
        
        return output_path
    except Exception as e:
        console.print(f"[red]Batch failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def run_all_batches(question_id: int, total_students: int, batch_size: int, delay_seconds: int = 2):
    """
    Run classifier test on multiple batches of students.
    
    Args:
        question_id: Question ID to test
        total_students: Total number of students to test
        batch_size: Number of students per batch
        delay_seconds: Delay between batches to avoid rate limits
    """
    # Calculate number of batches
    num_batches = (total_students + batch_size - 1) // batch_size
    
    console.print(f"\n[bold cyan]Starting Batch Classification Test[/bold cyan]")
    console.print(f"Question ID: {question_id}")
    console.print(f"Total Students: {total_students} (S0001 - S{total_students:04d})")
    console.print(f"Batch Size: {batch_size}")
    console.print(f"Number of Batches: {num_batches}")
    console.print(f"Delay Between Batches: {delay_seconds}s\n")
    
    successful_batches = 0
    failed_batches = 0
    batch_result_files = []  # Track all batch result files for merging
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"[cyan]Processing {num_batches} batches...",
            total=num_batches
        )
        
        for batch_num in range(num_batches):
            start_student_num = batch_num * batch_size + 1
            start_student_id = f"S{start_student_num:04d}"
            current_batch_size = min(batch_size, total_students - batch_num * batch_size)
            
            # Run batch
            batch_file = await run_batch(question_id, start_student_id, current_batch_size)
            
            if batch_file:
                successful_batches += 1
                batch_result_files.append(batch_file)  # Track successful batch files
            else:
                failed_batches += 1
            
            # Update progress
            progress.update(task, advance=1)
            
            # Delay between batches (except for the last one)
            if batch_num < num_batches - 1:
                console.print(f"[dim]Waiting {delay_seconds}s before next batch...[/dim]")
                await asyncio.sleep(delay_seconds)
    
    # Final summary
    console.print("\n[bold cyan]===== BATCH TEST COMPLETE =====[/bold cyan]")
    console.print(f"Total Batches: {num_batches}")
    console.print(f"[green]Successful: {successful_batches}[/green]")
    console.print(f"[red]Failed: {failed_batches}[/red]")
    console.print(f"Success Rate: {(successful_batches/num_batches*100):.1f}%\n")
    
    # Merge all batch results into a single file
    if batch_result_files:
        merged_file = merge_batch_results(question_id, batch_result_files, total_students)
        if merged_file:
            console.print(f"[bold green]✓ All results merged into: {merged_file}[/bold green]")
    else:
        console.print("[yellow]No batch results to merge[/yellow]")
    
    # Show where individual batch results are saved
    console.print(f"[dim]Individual batch results: tests/batch_results_Q{question_id}_S*.csv[/dim]")


def merge_batch_results(question_id: int, batch_files: list, total_students: int) -> str:
    """
    Merge all batch result CSV files into a single consolidated file.
    
    Args:
        question_id: Question ID
        batch_files: List of batch result file paths
        total_students: Total number of students processed
    
    Returns:
        Path to merged file
    """
    import csv
    from collections import Counter
    
    try:
        console.print(f"\n[cyan]Merging {len(batch_files)} batch result files...[/cyan]")
        
        # Generate merged filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = f"tests/merged_results_Q{question_id}_S0001-S{total_students:04d}_{timestamp}.csv"
        
        all_rows = []
        header = None
        
        # Read all batch CSV files
        for batch_file in batch_files:
            if Path(batch_file).exists():
                with open(batch_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    if header is None:
                        header = reader.fieldnames
                    
                    rows = list(reader)
                    all_rows.extend(rows)
                    console.print(f"[dim]  • Loaded {batch_file}: {len(rows)} rows[/dim]")
        
        if not all_rows:
            console.print("[red]No valid batch files to merge[/red]")
            return None
        
        # Sort by answer_id for consistency
        all_rows.sort(key=lambda x: int(x.get('answer_id', 0)))
        
        # Write merged file
        with open(merged_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)
        
        # Display summary statistics
        console.print(f"\n[bold cyan]Merged Results Summary[/bold cyan]")
        console.print(f"Total Records: {len(all_rows)}")
        
        # Count classifications
        if 'classification_label' in header:
            classifications = [row.get('classification_label', '') for row in all_rows]
            classification_counts = Counter(classifications)
            console.print(f"\n[bold]Classification Distribution:[/bold]")
            for label, count in classification_counts.most_common():
                if label:  # Skip empty labels
                    percentage = (count / len(all_rows)) * 100
                    console.print(f"  • {label.upper()}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        if 'classification_confidence' in header:
            confidences = [float(row.get('classification_confidence', 0)) for row in all_rows if row.get('classification_confidence')]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                console.print(f"\nAverage Confidence: {avg_confidence:.2f}")
        
        return merged_file
    
    except Exception as e:
        console.print(f"[red]Failed to merge batch results: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run classifier test in batches to test multiple students"
    )
    
    parser.add_argument(
        "--question-id",
        type=int,
        required=True,
        help="Question ID to test (1-4)"
    )
    
    parser.add_argument(
        "--total-students",
        type=int,
        default=50,
        help="Total number of students to test (default: 50)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of students per batch (default: 5)"
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay in seconds between batches (default: 2)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.total_students < 1:
        console.print("[red]Error: total-students must be at least 1[/red]")
        return
    
    if args.batch_size < 1:
        console.print("[red]Error: batch-size must be at least 1[/red]")
        return
    
    if args.delay < 0:
        console.print("[red]Error: delay must be non-negative[/red]")
        return
    
    # Run batches
    await run_all_batches(
        question_id=args.question_id,
        total_students=args.total_students,
        batch_size=args.batch_size,
        delay_seconds=args.delay
    )


if __name__ == "__main__":
    asyncio.run(main())
