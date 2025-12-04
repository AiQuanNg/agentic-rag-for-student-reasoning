"""
Test Extractor Agent on a batch of 10 student answers.

This CLI:
1. Fetches 10 student answers for a specific question
2. Processes them through the Extractor Agent
3. Displays real-time progress
4. Generates CSV results for analysis
5. Shows summary statistics

Usage:
    python cli/test_extractor.py --question-id 1 --limit 10
    python cli/test_extractor.py --question-id 2 --output results.csv
"""

import asyncio
import argparse
import logging
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncpg
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from dotenv import load_dotenv

# Import your modules
from agent.extractor import extract_keywords
from agent.config.providers import provider_manager
from agent.models.extraction import ExtractionResult

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_extractor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


class ExtractorTester:
    """Test the Extractor Agent on student answers."""
    
    def __init__(self, db_url: str):
        """Initialize tester with database connection."""
        self.db_url = db_url
        self.db_pool: Optional[asyncpg.Pool] = None
        self.results: List[dict] = []
    
    async def connect_db(self):
        """Create database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(self.db_url)
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def disconnect_db(self):
        """Close database connection pool."""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database disconnected")
    
    async def fetch_answers(
        self,
        question_id: int,
        limit: int = 10,
        student_ids: List[str] = None
    ) -> List[dict]:
        """
        Fetch student answers from database.
        
        Args:
            question_id: Question ID to fetch answers for
            limit: Maximum number of answers to fetch
            student_ids: Optional list of specific student IDs to fetch
            
        Returns:
            List of answer dictionaries
        """
        async with self.db_pool.acquire() as conn:
            if student_ids:
                # Fetch answers for specific students
                rows = await conn.fetch("""
                    SELECT 
                        id,
                        question_id,
                        student_id,
                        answer_text,
                        created_at
                    FROM student_submissions
                    WHERE question_id = $1 AND student_id = ANY($2)
                    ORDER BY student_id
                """, question_id, student_ids)
            else:
                # Fetch most recent answers
                rows = await conn.fetch("""
                    SELECT 
                        id,
                        question_id,
                        student_id,
                        answer_text,
                        created_at
                    FROM student_submissions
                    WHERE question_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, question_id, limit)
            
            return [dict(row) for row in rows]
    
    async def process_answer(
        self,
        answer_id: int,
        question_id: int,
        answer_text: str
    ) -> dict:
        """
        Process single answer through Extractor Agent.
        
        Args:
            answer_id: Student submission ID
            question_id: Question ID
            answer_text: Student's answer text
            
        Returns:
            Result dictionary with extraction and metadata
        """
        try:
            # Call extractor agent
            extraction = await extract_keywords(
                db_pool=self.db_pool,
                question_id=question_id,
                answer_text=answer_text
            )
            
            # Build result with metadata
            result = {
                'answer_id': answer_id,
                'question_id': question_id,
                'answer_text': answer_text,  # Add answer text
                'status': 'success',
                'topic': extraction.topic,  # New test field
                'matched_keywords': extraction.matched_keywords,
                'detected_themes': extraction.detected_themes,
                'novel_terms': extraction.novel_terms,
                'evidence_spans': extraction.evidence_spans,
                'extraction_confidence': extraction.extraction_confidence,
                'error': None
            }
            
            logger.info(
                f"Processed answer {answer_id}: "
                f"confidence={extraction.extraction_confidence:.2f}, "
                f"keywords={len(extraction.matched_keywords)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process answer {answer_id}: {e}")
            return {
                'answer_id': answer_id,
                'question_id': question_id,
                'answer_text': answer_text,  # Add answer text
                'status': 'error',
                'topic': [],  # New test field
                'matched_keywords': [],
                'detected_themes': [],
                'novel_terms': [],
                'evidence_spans': [],
                'extraction_confidence': 0.0,
                'error': str(e)
            }
    
    async def test_batch(
        self,
        question_id: int,
        limit: int = 10,
        student_ids: List[str] = None
    ) -> List[dict]:
        """
        Test Extractor on a batch of answers.
        
        Args:
            question_id: Question ID
            limit: Number of answers to test
            student_ids: Optional list of specific student IDs to test
            
        Returns:
            List of results
        """
        # Fetch answers
        if student_ids:
            console.print(
                f"\n[bold cyan]Fetching answers for {len(student_ids)} specific students "
                f"(Question {question_id})...[/bold cyan]"
            )
        else:
            console.print(
                f"\n[bold cyan]Fetching up to {limit} answers for Question {question_id}...[/bold cyan]"
            )
        
        answers = await self.fetch_answers(question_id, limit, student_ids)
        
        if not answers:
            console.print("[red]No answers found![/red]")
            return []
        
        console.print(f"[green]Found {len(answers)} answers[/green]\n")
        
        # Process with progress bar
        self.results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing answers...",
                total=len(answers)
            )
            
            # Process answers concurrently (up to 5 at a time for rate limiting)
            semaphore = asyncio.Semaphore(5)
            
            async def process_with_semaphore(answer):
                async with semaphore:
                    result = await self.process_answer(
                        answer['id'],
                        answer['question_id'],
                        answer['answer_text']
                    )
                    progress.update(task, advance=1)
                    return result
            
            # Run all in parallel
            self.results = await asyncio.gather(
                *[process_with_semaphore(answer) for answer in answers]
            )
        
        return self.results
    
    def display_results(self):
        """Display results in formatted tables."""
        
        if not self.results:
            console.print("[red]No results to display[/red]")
            return
        
        # Summary statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful
        avg_confidence = (
            sum(r['extraction_confidence'] for r in self.results if r['status'] == 'success') 
            / successful if successful > 0 else 0
        )
        
        summary_table = Table(title="Batch Processing Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Processed", str(total))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        summary_table.add_row("Average Confidence", f"{avg_confidence:.2f}")
        summary_table.add_row("Success Rate", f"{(successful/total*100):.1f}%")
        
        console.print("\n")
        console.print(summary_table)
        
        # Detailed results table
        results_table = Table(
            title="Detailed Extraction Results",
            show_header=True
        )
        results_table.add_column("ID", style="cyan")
        results_table.add_column("Status", style="magenta")
        results_table.add_column("Confidence", style="yellow")
        results_table.add_column("Topic", style="bright_magenta")  # New test field
        results_table.add_column("Keywords", style="green")
        results_table.add_column("Themes", style="blue")
        results_table.add_column("Novel Terms", style="red")
        
        for result in self.results:
            topic_count = len(result['topic'])  # New test field
            keyword_count = len(result['matched_keywords'])
            theme_count = len(result['detected_themes'])
            novel_count = len(result['novel_terms'])
            
            confidence_str = (
                f"{result['extraction_confidence']:.2f}" 
                if result['status'] == 'success' 
                else "ERROR"
            )
            
            results_table.add_row(
                str(result['answer_id']),
                result['status'].upper(),
                confidence_str,
                str(topic_count),  # New test field
                str(keyword_count),
                str(theme_count),
                str(novel_count)
            )
        
        console.print("\n")
        console.print(results_table)
        
        # Detailed view for each answer
        console.print("\n[bold cyan]DETAILED EXTRACTION RESULTS[/bold cyan]\n")
        
        for i, result in enumerate(self.results, 1):
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            
            console.print(
                f"\n[bold]{status_symbol} Answer {i} (ID: {result['answer_id']})[/bold]"
            )
            console.print(f"  Confidence: {result['extraction_confidence']:.2f}")
            
            # Display answer text (truncated)
            if result['answer_text']:
                answer_preview = result['answer_text'][:100] + "..." if len(result['answer_text']) > 100 else result['answer_text']
                console.print(f"  [dim]Answer:[/dim] {answer_preview}")
            
            # Display topics
            if result['topic']:
                console.print(f"  [bright_magenta]Topics ({len(result['topic'])}):[/bright_magenta] {', '.join(result['topic'])}")
            
            if result['matched_keywords']:
                console.print(f"  Keywords: {', '.join(result['matched_keywords'][:5])}")
            
            if result['detected_themes']:
                console.print(f"  Themes: {', '.join(result['detected_themes'])}")
            
            if result['novel_terms']:
                console.print(f"  [yellow]Novel Terms:[/yellow] {', '.join(result['novel_terms'])}")
            
            if result['error']:
                console.print(f"  [red]Error: {result['error']}[/red]")
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save results to CSV file.
        
        Args:
            output_path: Output file path (default: results_Q{question_id}_{timestamp}.csv)
            
        Returns:
            Path to saved CSV file
        """
        if not self.results:
            logger.warning("No results to save")
            return None
        
        # Generate filename if not provided
        if not output_path:
            question_id = self.results[0]['question_id']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"tests/test_results_Q{question_id}_{timestamp}.csv"
        
        # Write CSV
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'answer_id',
                    'question_id',
                    'answer_text',  # Add answer text
                    'status',
                    'extraction_confidence',
                    'topic_count',  # Add topic count
                    'topic',  # New test field
                    'matched_keywords_count',
                    'detected_themes_count',
                    'novel_terms_count',
                    'evidence_spans_count',
                    'matched_keywords',
                    'detected_themes',
                    'novel_terms',
                    'evidence_spans',
                    'error'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'answer_id': result['answer_id'],
                        'question_id': result['question_id'],
                        'answer_text': result['answer_text'],  # Add answer text
                        'status': result['status'],
                        'extraction_confidence': result['extraction_confidence'],
                        'topic_count': len(result['topic']),  # Add topic count
                        'topic': '|'.join(result['topic']),  # New test field
                        'matched_keywords_count': len(result['matched_keywords']),
                        'detected_themes_count': len(result['detected_themes']),
                        'novel_terms_count': len(result['novel_terms']),
                        'evidence_spans_count': len(result['evidence_spans']),
                        'matched_keywords': '|'.join(result['matched_keywords']),
                        'detected_themes': '|'.join(result['detected_themes']),
                        'novel_terms': '|'.join(result['novel_terms']),
                        'evidence_spans': '|'.join(result['evidence_spans']),
                        'error': result['error'] or ''
                    }
                    writer.writerow(row)
            
            logger.info(f"Results saved to {output_path}")
            console.print(f"\n[green]Results saved to: {output_path}[/green]")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            console.print(f"[red]Failed to save results: {e}[/red]")
            return None
    
    def print_model_info(self):
        """Print information about the configured LLM model."""
        try:
            model_info = provider_manager.get_model_info("extractor")
            
            info_panel = Panel(
                f"[bold]Provider:[/bold] {model_info['provider']}\n"
                f"[bold]Model:[/bold] {model_info['model']}\n"
                f"[bold]Temperature:[/bold] {model_info['temperature']}\n"
                f"[bold]Cost:[/bold] {model_info['cost']}",
                title="Extractor Agent Configuration"
            )
            console.print(info_panel)
            
        except Exception as e:
            console.print(f"[red]Failed to get model info: {e}[/red]")


async def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Test Extractor Agent on student answers"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        required=True,
        help="Question ID to test"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of answers to test (default: 10)"
    )
    parser.add_argument(
        "--students",
        type=str,
        default=None,
        help="Comma-separated list of student IDs (e.g., S0001,S0002,S0003) or range (e.g., S0001-S0010)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Database URL (uses DATABASE_URL env var if not provided)"
    )
    
    args = parser.parse_args()
    
    # Parse student IDs if provided
    student_ids = None
    if args.students:
        if '-' in args.students and ',' not in args.students:
            # Range format: S0001-S0010
            start, end = args.students.split('-')
            start_num = int(start[1:])  # Remove 'S' prefix
            end_num = int(end[1:])
            student_ids = [f"S{i:04d}" for i in range(start_num, end_num + 1)]
        else:
            # Comma-separated format: S0001,S0002,S0003
            student_ids = [s.strip() for s in args.students.split(',')]
        
        console.print(f"[yellow]Testing specific students: {', '.join(student_ids)}[/yellow]")
    
    # Get database URL
    import os
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        console.print("[red]Error: DATABASE_URL not set[/red]")
        return
    
    # Initialize tester
    tester = ExtractorTester(db_url)
    
    try:
        # Connect to database
        await tester.connect_db()
        
        # Display configuration
        console.print("[bold cyan]=== EXTRACTOR AGENT TEST ===[/bold cyan]")
        tester.print_model_info()
        
        # Run tests
        console.print(
            f"\n[bold]Testing Extractor on Question {args.question_id}[/bold]"
        )
        
        results = await tester.test_batch(args.question_id, args.limit, student_ids)
        
        if results:
            # Display results
            tester.display_results()
            
            # Save results
            output_file = tester.save_results(args.output)
            
            console.print("\n[bold green]Test completed successfully![/bold green]")
        else:
            console.print("[red]No results to display[/red]")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        console.print(f"\n[red]Error: {e}[/red]")
        
    finally:
        # Cleanup
        await tester.disconnect_db()


if __name__ == "__main__":
    asyncio.run(main())
