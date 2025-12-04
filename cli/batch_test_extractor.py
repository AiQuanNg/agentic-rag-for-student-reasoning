"""
Batch Test Extractor - Run extractor on multiple batches of students.

This script runs the extractor test in batches to avoid rate limits
and provide progress updates.

Usage:
    python cli/batch_test_extractor.py --question-id 1 --total-students 30 --batch-size 5
    python cli/batch_test_extractor.py --question-id 1 --start-student 1001 --total-students 32 --batch-size 5
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

# Import ExtractorTester class
from cli.test_extractor import ExtractorTester

load_dotenv()
console = Console()


async def run_batch(question_id: int, start_student_num: int, batch_size: int, output_dir: str = "tests/batches") -> tuple:
    """
    Run a single batch of student extraction tests.
    
    Args:
        question_id: Question ID to test
        start_student_num: Starting student number (e.g., 1 for S0001, 1001 for S1001)
        batch_size: Number of students in this batch
        output_dir: Directory to save batch results
    
    Returns:
        Tuple of (output_file_path, success_count, total_count)
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate list of student IDs for this batch
        student_ids = [f"S{i:04d}" for i in range(start_student_num, start_student_num + batch_size)]
        student_range = f"S{start_student_num:04d}-S{start_student_num + batch_size - 1:04d}"
        
        console.print(f"\n[cyan]Processing batch: {student_range}[/cyan]")
        
        # Get database URL
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            console.print("[red]DATABASE_URL not found in environment[/red]")
            return None, 0, batch_size
        
        # Create tester instance and run extraction
        tester = ExtractorTester(db_url)
        await tester.connect_db()
        
        try:
            # Run batch test
            results = await tester.test_batch(
                question_id=question_id,
                limit=batch_size,
                student_ids=student_ids
            )
            
            if not results:
                console.print(f"[yellow]No results for batch {student_range}[/yellow]")
                return None, 0, batch_size
            
            # Generate batch-specific output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_dir}/batch_Q{question_id}_{student_range}_{timestamp}.csv"
            
            # Save results using the tester's save_results method
            saved_path = tester.save_results(output_file)
            
            # Count successful extractions (confidence > 0)
            success_count = sum(1 for r in results if r.get('extraction_confidence', 0) > 0)
            
            console.print(f"[green]✓ Batch saved: {saved_path}[/green]")
            console.print(f"[dim]  Success: {success_count}/{len(results)}[/dim]")
            
            return saved_path, success_count, len(results)
            
        finally:
            await tester.disconnect_db()
        
    except Exception as e:
        console.print(f"[red]Batch failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None, 0, batch_size


async def run_all_batches(
    question_id: int,
    total_students: int,
    batch_size: int,
    delay_seconds: int = 3,
    start_student: int = 1
):
    """
    Run extractor test on multiple batches of students.
    
    Args:
        question_id: Question ID to test
        total_students: Total number of students to test
        batch_size: Number of students per batch
        delay_seconds: Delay between batches to avoid rate limits
        start_student: Starting student number (default: 1 for S0001)
    """
    # Calculate number of batches
    num_batches = (total_students + batch_size - 1) // batch_size
    end_student = start_student + total_students - 1
    
    console.print(f"\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║   Batch Extraction Test - Starting      ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")
    console.print(f"\n📋 [bold]Configuration:[/bold]")
    console.print(f"   • Question ID: {question_id}")
    console.print(f"   • Students: S{start_student:04d} - S{end_student:04d} ({total_students} total)")
    console.print(f"   • Batch Size: {batch_size} students")
    console.print(f"   • Total Batches: {num_batches}")
    console.print(f"   • Delay: {delay_seconds}s between batches\n")
    
    successful_batches = 0
    failed_batches = 0
    batch_result_files = []
    total_success = 0
    total_processed = 0
    
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
            start_student_num = start_student + batch_num * batch_size
            current_batch_size = min(batch_size, total_students - batch_num * batch_size)
            
            console.print(f"\n[bold cyan]>>> Batch {batch_num + 1}/{num_batches}[/bold cyan]")
            
            # Run batch
            batch_file, success_count, processed_count = await run_batch(
                question_id=question_id,
                start_student_num=start_student_num,
                batch_size=current_batch_size
            )
            
            if batch_file:
                successful_batches += 1
                batch_result_files.append(batch_file)
                total_success += success_count
                total_processed += processed_count
            else:
                failed_batches += 1
                total_processed += current_batch_size
            
            # Update progress
            progress.update(task, advance=1)
            
            # Delay between batches (except for the last one)
            if batch_num < num_batches - 1:
                console.print(f"[dim]⏳ Waiting {delay_seconds}s before next batch...[/dim]")
                await asyncio.sleep(delay_seconds)
    
    # Final summary
    console.print(f"\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print(f"[bold cyan]║      Batch Test Complete - Summary      ║[/bold cyan]")
    console.print(f"[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]\n")
    
    console.print(f"📊 [bold]Batch Statistics:[/bold]")
    console.print(f"   • Total Batches: {num_batches}")
    console.print(f"   • [green]Successful: {successful_batches}[/green]")
    console.print(f"   • [red]Failed: {failed_batches}[/red]")
    console.print(f"   • Success Rate: {(successful_batches/num_batches*100):.1f}%")
    
    console.print(f"\n📈 [bold]Extraction Statistics:[/bold]")
    console.print(f"   • Total Students: {total_processed}")
    console.print(f"   • [green]Successful Extractions: {total_success}[/green]")
    console.print(f"   • [yellow]Empty Results: {total_processed - total_success}[/yellow]")
    console.print(f"   • Extraction Rate: {(total_success/total_processed*100):.1f}%\n")
    
    # Merge all batch results into a single file
    if batch_result_files:
        merged_file = merge_batch_results(
            question_id=question_id,
            batch_files=batch_result_files,
            start_student=start_student,
            end_student=end_student
        )
        if merged_file:
            console.print(f"\n[bold green]✓ All results merged successfully![/bold green]")
            console.print(f"[bold green]📁 Merged file: {merged_file}[/bold green]")
    else:
        console.print("\n[yellow]⚠ No batch results to merge[/yellow]")
    
    # Show where files are saved
    console.print(f"\n[dim]📂 Individual batches: tests/batches/batch_Q{question_id}_S*.csv[/dim]")


def merge_batch_results(
    question_id: int,
    batch_files: list,
    start_student: int,
    end_student: int
) -> str:
    """
    Merge all batch result CSV files into a single consolidated file.
    
    Args:
        question_id: Question ID
        batch_files: List of batch result file paths
        start_student: Starting student number
        end_student: Ending student number
    
    Returns:
        Path to merged file
    """
    import csv
    from collections import Counter
    
    try:
        console.print(f"\n[cyan]🔄 Merging {len(batch_files)} batch result files...[/cyan]")
        
        # Generate merged filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = f"tests/extractor_Q{question_id}_S{start_student:04d}-S{end_student:04d}_{timestamp}.csv"
        
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
                    console.print(f"[dim]  ✓ Loaded {Path(batch_file).name}: {len(rows)} rows[/dim]")
        
        if not all_rows:
            console.print("[red]✗ No valid batch files to merge[/red]")
            return None
        
        # Sort by answer_id for consistency
        all_rows.sort(key=lambda x: int(x.get('answer_id', 0)))
        
        # Write merged file
        with open(merged_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_rows)
        
        # Display summary statistics
        console.print(f"\n[bold cyan]📊 Merged Results Summary[/bold cyan]")
        console.print(f"   • Total Records: {len(all_rows)}")
        
        # Count successful extractions
        successful = sum(1 for row in all_rows if float(row.get('extraction_confidence', 0)) > 0)
        console.print(f"   • Successful Extractions: {successful} ({(successful/len(all_rows)*100):.1f}%)")
        
        # Average confidence
        if 'extraction_confidence' in header:
            confidences = [float(row.get('extraction_confidence', 0)) for row in all_rows]
            avg_confidence = sum(confidences) / len(confidences)
            console.print(f"   • Average Confidence: {avg_confidence:.2f}")
        
        # Topic statistics
        if 'topic' in header:
            topic_counts = []
            for row in all_rows:
                topics = row.get('topic', '').split('|') if row.get('topic') else []
                topics = [t for t in topics if t.strip()]  # Filter empty
                topic_counts.append(len(topics))
            
            if topic_counts:
                avg_topics = sum(topic_counts) / len(topic_counts)
                console.print(f"   • Average Topics per Answer: {avg_topics:.2f}")
                console.print(f"   • Max Topics in Answer: {max(topic_counts)}")
        
        # Keyword statistics
        if 'matched_keywords' in header:
            keyword_counts = []
            for row in all_rows:
                keywords = row.get('matched_keywords', '').split('|') if row.get('matched_keywords') else []
                keywords = [k for k in keywords if k.strip()]
                keyword_counts.append(len(keywords))
            
            if keyword_counts:
                avg_keywords = sum(keyword_counts) / len(keyword_counts)
                console.print(f"   • Average Keywords per Answer: {avg_keywords:.2f}")
        
        return merged_file
    
    except Exception as e:
        console.print(f"[red]Failed to merge batch results: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run extractor test in batches to test multiple students",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 30 students (S0001-S0030) in batches of 5
  python cli/batch_test_extractor.py --question-id 1 --total-students 30 --batch-size 5
  
  # Test students S1001-S1032 in batches of 5
  python cli/batch_test_extractor.py --question-id 1 --start-student 1001 --total-students 32 --batch-size 5
  
  # Test with custom delay (useful for rate limiting)
  python cli/batch_test_extractor.py --question-id 2 --total-students 20 --batch-size 3 --delay 5
        """
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
        required=True,
        help="Total number of students to test (e.g., 30)"
    )
    
    parser.add_argument(
        "--start-student",
        type=int,
        default=1,
        help="Starting student number (default: 1 for S0001)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of students per batch (default: 5, recommended: 3-5)"
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=3,
        help="Delay in seconds between batches (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.total_students < 1:
        console.print("[red]Error: total-students must be at least 1[/red]")
        return
    
    if args.batch_size < 1:
        console.print("[red]Error: batch-size must be at least 1[/red]")
        return
    
    if args.batch_size > 10:
        console.print("[yellow]Warning: batch-size > 10 may hit rate limits[/yellow]")
    
    if args.delay < 0:
        console.print("[red]Error: delay must be non-negative[/red]")
        return
    
    if args.question_id not in [1, 2, 3, 4]:
        console.print("[red]Error: question-id must be 1, 2, 3, or 4[/red]")
        return
    
    # Run batches
    start_time = time.time()
    await run_all_batches(
        question_id=args.question_id,
        total_students=args.total_students,
        batch_size=args.batch_size,
        delay_seconds=args.delay,
        start_student=args.start_student
    )
    end_time = time.time()
    
    # Show total time
    total_time = end_time - start_time
    console.print(f"\n⏱️  [bold]Total Time:[/bold] {total_time:.1f}s ({total_time/60:.1f} minutes)")


if __name__ == "__main__":
    asyncio.run(main())