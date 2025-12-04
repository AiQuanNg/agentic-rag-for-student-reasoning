"""
Test Classifier Agent on a batch of 10 student answers.

This CLI:
1. Fetches student answers for a specific question
2. Processes them through Extractor Agent (if not already extracted)
3. Classifies them through Classifier Agent (primary task)
4. Displays real-time progress with Rich UI
5. Generates CSV results for analysis
6. Shows summary statistics and classification distribution

Usage:
    python cli/test_classifier.py --question-id 1 --limit 10
    python cli/test_classifier.py --question-id 2 --output classification_results.csv
    python cli/test_classifier.py --question-id 1 --students S0001,S0002,S0003

"""

import asyncio
import argparse
import logging
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncpg
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

# Import your modules
from agent.extractor import extract_keywords
from agent.classifier import classify_answer
from agent.config.providers import provider_manager
from agent.models.extraction import ExtractionResult
from agent.models.classification import ClassificationResult

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_classifier.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


class ClassifierTester:
    """Test the Classifier Agent on student answers with extraction pipeline."""
    
    def __init__(self, db_url: str):
        """
        Initialize tester with database connection.
        
        Args:
            db_url: PostgreSQL database URL
        """
        self.db_url = db_url
        self.db_pool: Optional[asyncpg.Pool] = None
        self.results: List[Dict[str, Any]] = []
        self.extractor_cache: Dict[int, ExtractionResult] = {}  # Cache for loaded extractor results
    
    def load_extractor_results_from_csv(self, csv_path: str) -> Dict[int, ExtractionResult]:
        """
        Load extractor results from CSV file.
        
        Args:
            csv_path: Path to extractor results CSV file
            
        Returns:
            Dictionary mapping answer_id to ExtractionResult
        """
        import csv
        
        extractor_results = {}
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    answer_id = int(row['answer_id'])
                    
                    # Parse lists from pipe-separated strings
                    matched_keywords = row.get('matched_keywords', '').split('|') if row.get('matched_keywords') else []
                    detected_themes = row.get('detected_themes', '').split('|') if row.get('detected_themes') else []
                    novel_terms = row.get('novel_terms', '').split('|') if row.get('novel_terms') else []
                    topic = row.get('topic', '').split('|') if row.get('topic') else []
                    evidence_spans = row.get('evidence_spans', '').split('|') if row.get('evidence_spans') else []
                    
                    # Create ExtractionResult object
                    extraction = ExtractionResult(
                        topic=topic,
                        matched_keywords=matched_keywords,
                        detected_themes=detected_themes,
                        novel_terms=novel_terms,
                        extraction_confidence=float(row.get('extraction_confidence', 0.0)),
                        evidence_spans=evidence_spans,
                        reasoning=f"Loaded from CSV: {csv_path}"
                    )
                    
                    extractor_results[answer_id] = extraction
                    
            logger.info(f"Loaded {len(extractor_results)} extractor results from {csv_path}")
            console.print(f"[green]✓ Loaded {len(extractor_results)} extractor results from CSV[/green]")
            
            self.extractor_cache = extractor_results
            return extractor_results
            
        except Exception as e:
            logger.error(f"Failed to load extractor results from CSV: {e}")
            console.print(f"[red]✗ Failed to load extractor results: {e}[/red]")
            raise
    
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
    ) -> List[Dict[str, Any]]:
        """
        Fetch student answers from database.
        
        Args:
            question_id: Question ID to fetch answers for
            limit: Maximum number of answers to fetch
            student_ids: Optional list of specific student IDs to fetch
        
        Returns:
            List of answer dictionaries with id, question_id, student_id, answer_text
        """
        async with self.db_pool.acquire() as conn:
            if student_ids:
                # Fetch answers for specific students
                rows = await conn.fetch("""
                    SELECT id, question_id, student_id, answer_text, created_at
                    FROM student_submissions
                    WHERE question_id = $1 AND student_id = ANY($2)
                    ORDER BY student_id
                """, question_id, student_ids)
            else:
                # Fetch most recent answers
                rows = await conn.fetch("""
                    SELECT id, question_id, student_id, answer_text, created_at
                    FROM student_submissions
                    WHERE question_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, question_id, limit)
        
        return [dict(row) for row in rows]
    
    async def fetch_question_context(self, question_id: int) -> Dict[str, Any]:
        """
        Fetch question text from database.
        
        Args:
            question_id: Question ID
        
        Returns:
            Dictionary with question_text (LLM infers goal/topic from this)
        """
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT text as question_text
                    FROM questions
                    WHERE id = $1
                """, question_id)
            
            if not row:
                return {'question_text': None}
            
            return dict(row)
        except Exception as e:
            logger.warning(f"Failed to fetch question context: {e}")
            return {'question_text': None}
    
    async def process_answer(
        self,
        answer_id: int,
        question_id: int,
        answer_text: str,
        question_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process single answer through Extractor and Classifier pipeline.
        
        Pipeline:
        1. Extract keywords/themes from answer (via Extractor Agent OR load from cache)
        2. Pass extraction results to Classifier Agent
        3. Classify as STANDARD, LATENT, or OFF_TOPIC
        4. Return combined result
        
        Args:
            answer_id: Student submission ID
            question_id: Question ID (1-4)
            answer_text: Student's full answer
            question_context: Question metadata (text, goal, topic)
        
        Returns:
            Result dictionary with extraction + classification data
        """
        try:
            # STEP 1: Extract keywords/themes (via Extractor Agent OR load from cache)
            if answer_id in self.extractor_cache:
                # Use cached extractor results from CSV
                logger.info(f"Using cached extraction for answer {answer_id}...")
                extraction = self.extractor_cache[answer_id]
            else:
                # Run extractor agent
                logger.info(f"Extracting from answer {answer_id}...")
                extraction = await extract_keywords(
                    db_pool=self.db_pool,
                    question_id=question_id,
                    answer_text=answer_text
                )
            
            # STEP 2: Classify the answer (via Classifier Agent)
            logger.info(f"Classifying answer {answer_id}...")
            classification = await classify_answer(
                db_pool=self.db_pool,
                question_id=question_id,
                answer_text=answer_text,
                extraction_results=extraction,
                question_text=question_context.get('question_text')
            )
            
            # STEP 3: Build combined result with NEW classifier output structure
            result = {
                # Metadata
                'answer_id': answer_id,
                'question_id': question_id,
                'answer_text': answer_text,
                'status': 'success',
                
                # Extraction results (from Extractor Agent)
                'topic': extraction.topic,
                'matched_keywords': extraction.matched_keywords,
                'detected_themes': extraction.detected_themes,
                'novel_terms': extraction.novel_terms,
                'extraction_confidence': extraction.extraction_confidence,
                
                # Classification results (from Classifier Agent) - PRIMARY
                'classification_label': classification.label,
                'classification_confidence': classification.classification_confidence,
                'reasoning': classification.reasoning,
                'evidence_spans': classification.evidence_spans,
                
                # NEW: Rubric assessment (Level 0/50/100)
                'rubric_level_achieved': classification.rubric_assessment.get('rubric_level_achieved', 'unknown') if hasattr(classification, 'rubric_assessment') and classification.rubric_assessment else 'unknown',
                'rubric_level_100': classification.rubric_assessment.get('level_100_met', False) if hasattr(classification, 'rubric_assessment') and classification.rubric_assessment else False,
                'rubric_level_50': classification.rubric_assessment.get('level_50_met', False) if hasattr(classification, 'rubric_assessment') and classification.rubric_assessment else False,
                'rubric_level_0': classification.rubric_assessment.get('level_0_met', False) if hasattr(classification, 'rubric_assessment') and classification.rubric_assessment else False,
                
                # NEW: Flagged novel terms with importance scores
                'flagged_novel_terms_count': len(classification.flagged_novel_terms) if hasattr(classification, 'flagged_novel_terms') and classification.flagged_novel_terms else 0,
                'high_priority_terms': [t.get('term', '') for t in classification.flagged_novel_terms if t.get('priority') == 'HIGH'] if hasattr(classification, 'flagged_novel_terms') and classification.flagged_novel_terms else [],
                'medium_priority_terms': [t.get('term', '') for t in classification.flagged_novel_terms if t.get('priority') == 'MEDIUM'] if hasattr(classification, 'flagged_novel_terms') and classification.flagged_novel_terms else [],
                
                # NEW: Latent signals summary
                'latent_mechanism_explanations': classification.latent_signals_summary.get('mechanism_explanations', []) if hasattr(classification, 'latent_signals_summary') and classification.latent_signals_summary else [],
                'latent_novel_terms_in_mechanisms': classification.latent_signals_summary.get('novel_terms_in_mechanisms', []) if hasattr(classification, 'latent_signals_summary') and classification.latent_signals_summary else [],
                'latent_critical_engagement': bool(classification.latent_signals_summary.get('critical_engagement')) if hasattr(classification, 'latent_signals_summary') and classification.latent_signals_summary else False,
                
                # Aggregator recommendation (updated structure)
                'aggregator_recommendation': classification.aggregator_recommendation.get('route_to_aggregator', False) if hasattr(classification, 'aggregator_recommendation') and isinstance(classification.aggregator_recommendation, dict) else (classification.aggregator_recommendation if hasattr(classification, 'aggregator_recommendation') else False),
                'aggregator_reason': classification.aggregator_recommendation.get('reason', '') if hasattr(classification, 'aggregator_recommendation') and isinstance(classification.aggregator_recommendation, dict) else '',
                
                # Error tracking
                'error': None
            }
            
            logger.info(
                f"Processed answer {answer_id}: "
                f"label={classification.label}, "
                f"confidence={classification.classification_confidence:.2f}"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to process answer {answer_id}: {e}")
            return {
                'answer_id': answer_id,
                'question_id': question_id,
                'answer_text': answer_text,
                'status': 'error',
                'topic': [],
                'matched_keywords': [],
                'detected_themes': [],
                'novel_terms': [],
                'extraction_confidence': 0.0,
                'classification_label': 'error',
                'classification_confidence': 0.0,
                'reasoning': '',
                'evidence_spans': [],
                'rubric_level_achieved': 'unknown',
                'rubric_level_100': False,
                'rubric_level_50': False,
                'rubric_level_0': False,
                'flagged_novel_terms_count': 0,
                'high_priority_terms': [],
                'medium_priority_terms': [],
                'latent_mechanism_explanations': [],
                'latent_novel_terms_in_mechanisms': [],
                'latent_critical_engagement': False,
                'aggregator_recommendation': False,
                'aggregator_reason': 'Error',
                'error': str(e)
            }
    
    async def test_batch(
        self,
        question_id: int,
        limit: int = 10,
        student_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Test Classifier on a batch of answers with full pipeline.
        
        Pipeline:
        1. Fetch student answers (from DB)
        2. Fetch question context (for calibration)
        3. Process each answer through Extractor → Classifier
        4. Display progress in real-time
        5. Collect results
        
        Args:
            question_id: Question ID
            limit: Number of answers to test
            student_ids: Optional list of specific student IDs to test
        
        Returns:
            List of result dictionaries
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
        
        # Fetch question context for calibration
        console.print("[cyan]Fetching question context...[/cyan]")
        question_context = await self.fetch_question_context(question_id)
        if question_context.get('question_text'):
            console.print(f"[green]Question: {question_context['question_text'][:60]}...[/green]")
        
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
                "[cyan]Processing & Classifying answers...",
                total=len(answers)
            )
            
            # Process answers concurrently (up to 3 at a time for rate limiting)
            semaphore = asyncio.Semaphore(3)
            
            async def process_with_semaphore(answer):
                async with semaphore:
                    result = await self.process_answer(
                        answer['id'],
                        answer['question_id'],
                        answer['answer_text'],
                        question_context
                    )
                    progress.update(task, advance=1)
                    return result
            
            # Run all in parallel with concurrency limit
            self.results = await asyncio.gather(
                *[process_with_semaphore(answer) for answer in answers]
            )
        
        return self.results
    
    def display_results(self):
        """Display results in formatted tables with classification statistics."""
        if not self.results:
            console.print("[red]No results to display[/red]")
            return
        
        # ===== SUMMARY STATISTICS =====
        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        failed = total - successful
        
        # Classification distribution
        classifications = {}
        for r in self.results:
            if r['status'] == 'success':
                label = r['classification_label']
                classifications[label] = classifications.get(label, 0) + 1
        
        # Confidence statistics (for successful classifications)
        successful_results = [r for r in self.results if r['status'] == 'success']
        if successful_results:
            avg_confidence = sum(r['classification_confidence'] for r in successful_results) / len(successful_results)
            min_confidence = min(r['classification_confidence'] for r in successful_results)
            max_confidence = max(r['classification_confidence'] for r in successful_results)
        else:
            avg_confidence = min_confidence = max_confidence = 0.0
        
        # Aggregator routing
        route_to_aggregator = sum(1 for r in self.results if r.get('aggregator_recommendation') == True or r.get('aggregator_recommendation') == 'ROUTE')
        
        # Summary table
        summary_table = Table(title="Batch Classification Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Processed", str(total))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        summary_table.add_row("Success Rate", f"{(successful/total*100):.1f}%")
        summary_table.add_row("Avg Confidence", f"{avg_confidence:.2f}")
        summary_table.add_row("Min/Max Confidence", f"{min_confidence:.2f} / {max_confidence:.2f}")
        summary_table.add_row("Route to Aggregator", str(route_to_aggregator))
        
        console.print("\n")
        console.print(summary_table)
        
        # ===== CLASSIFICATION DISTRIBUTION =====
        if classifications:
            dist_table = Table(title="Classification Distribution", show_header=True)
            dist_table.add_column("Label", style="cyan")
            dist_table.add_column("Count", style="green")
            dist_table.add_column("Percentage", style="yellow")
            
            for label in ['standard', 'latent', 'off_topic', 'error']:
                if label in classifications:
                    count = classifications[label]
                    pct = (count / successful * 100) if successful > 0 else 0
                    dist_table.add_row(label.upper(), str(count), f"{pct:.1f}%")
            
            console.print("\n")
            console.print(dist_table)
        
        # ===== DETAILED RESULTS TABLE =====
        results_table = Table(
            title="Detailed Classification Results",
            show_header=True
        )
        
        results_table.add_column("ID", style="cyan", width=5)
        results_table.add_column("Status", style="magenta", width=8)
        results_table.add_column("Label", style="yellow", width=10)
        results_table.add_column("Conf", style="green", width=5)
        results_table.add_column("Rubric", style="blue", width=6)
        results_table.add_column("Topics", style="bright_magenta", width=7)
        results_table.add_column("Keywords", style="blue", width=8)
        results_table.add_column("Novel\nTerms", style="red", width=8)
        results_table.add_column("Flagged\n(H/M)", style="yellow", width=8)
        results_table.add_column("Agg", style="magenta", width=8)
        
        for result in self.results:
            if result['status'] == 'success':
                # Color-code label
                label = result['classification_label'].upper()
                if label == 'STANDARD':
                    label_text = Text(label, style="blue")
                elif label == 'LATENT':
                    label_text = Text(label, style="magenta")
                else:
                    label_text = Text(label, style="red")
                
                # Latent signals indicator
                high_priority = len(result.get('high_priority_terms', []))
                medium_priority = len(result.get('medium_priority_terms', []))
                
                # Aggregator recommendation
                agg_rec = result.get('aggregator_recommendation')
                if agg_rec == True or agg_rec == 'ROUTE':
                    agg_text = "ROUTE"
                    agg_style = "magenta"
                elif agg_rec == False or agg_rec == 'BASELINE':
                    agg_text = "BASELINE"
                    agg_style = "green"
                else:
                    agg_text = str(agg_rec) if agg_rec else "BASELINE"
                    agg_style = "green"
                
                results_table.add_row(
                    str(result['answer_id']),
                    result['status'].upper(),
                    label_text,
                    f"{result['classification_confidence']:.2f}",
                    result.get('rubric_level_achieved', 'unknown'),
                    str(len(result['topic'])),
                    str(len(result['matched_keywords'])),
                    str(len(result['novel_terms'])),
                    f"{high_priority}/{medium_priority}",
                    Text(agg_text, style=agg_style)
                )
            else:
                results_table.add_row(
                    str(result['answer_id']),
                    "ERROR",
                    "N/A",
                    "0.00",
                    "unknown",
                    "0",
                    "0",
                    "0",
                    "0/0",
                    "BASELINE"
                )
        
        console.print("\n")
        console.print(results_table)
        
        # ===== LATENT DISCOVERIES (Key research findings) =====
        latent_results = [r for r in self.results if r['status'] == 'success' and r['classification_label'] == 'latent']
        if latent_results:
            console.print("\n[bold cyan]=== LATENT DISCOVERIES (Research Findings) ===[/bold cyan]")
            
            high_conf_latent = [r for r in latent_results if r['classification_confidence'] >= 0.75]
            route_latent = [r for r in latent_results if r.get('aggregator_recommendation') == True or r.get('aggregator_recommendation') == 'ROUTE']
            
            console.print(f"\nHigh-Confidence LATENT (≥0.75): {len(high_conf_latent)}")
            for r in high_conf_latent[:5]:  # Show top 5
                console.print(
                    f"  • Answer {r['answer_id']}: "
                    f"confidence={r['classification_confidence']:.2f}, "
                    f"rubric={r.get('rubric_level_achieved', 'unknown')}, "
                    f"flagged_terms={len(r.get('high_priority_terms', []))}H/{len(r.get('medium_priority_terms', []))}M"
                )
            
            console.print(f"\nEmerging LATENT Insights (→ Aggregator): {len(route_latent)}")
            for r in route_latent[:5]:  # Show top 5
                console.print(
                    f"  • Answer {r['answer_id']}: "
                    f"confidence={r['classification_confidence']:.2f}, "
                    f"reason={r.get('aggregator_reason', 'Unknown')}"
                )
    
    def save_results(self, output_path: str = None) -> Optional[str]:
        """
        Save results to CSV file.
        
        CSV structure:
        - Metadata: answer_id, question_id, status
        - Extraction: matched_keywords, detected_themes, novel_terms, extraction_confidence
        - Classification: label, confidence, reasoning, evidence
        - Aggregator: recommendation
        
        Args:
            output_path: Output file path (default: test_results_Q{question_id}_{timestamp}.csv)
        
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
            output_path = f"tests/test_results_classifier_Q{question_id}_{timestamp}.csv"
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    # Metadata
                    'answer_id',
                    'question_id',
                    'status',
                    'answer_text',
                    
                    # Extraction results
                    'extraction_confidence',
                    'topic_count',
                    'topic',
                    'matched_keywords_count',
                    'detected_themes_count',
                    'novel_terms_count',
                    'matched_keywords',
                    'detected_themes',
                    'novel_terms',
                    
                    # Classification results (PRIMARY)
                    'classification_label',
                    'classification_confidence',
                    'reasoning',
                    'evidence_spans_count',
                    'evidence_spans',
                    
                    # Rubric assessment (NEW)
                    'rubric_level_achieved',
                    'rubric_level_100',
                    'rubric_level_50',
                    'rubric_level_0',
                    
                    # Novel term flagging (NEW)
                    'flagged_novel_terms_count',
                    'high_priority_terms',
                    'medium_priority_terms',
                    
                    # Latent signals (NEW)
                    'latent_mechanism_explanations',
                    'latent_novel_terms_in_mechanisms',
                    'latent_critical_engagement',
                    
                    # Aggregator routing (UPDATED)
                    'aggregator_recommendation',
                    'aggregator_reason',
                    
                    # Error tracking
                    'error'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        # Metadata
                        'answer_id': result['answer_id'],
                        'question_id': result['question_id'],
                        'status': result['status'],
                        'answer_text': result['answer_text'],
                        
                        # Extraction results
                        'extraction_confidence': result['extraction_confidence'],
                        'topic_count': len(result['topic']),
                        'topic': '|'.join(result['topic']),
                        'matched_keywords_count': len(result['matched_keywords']),
                        'detected_themes_count': len(result['detected_themes']),
                        'novel_terms_count': len(result['novel_terms']),
                        'matched_keywords': '|'.join(result['matched_keywords']),
                        'detected_themes': '|'.join(result['detected_themes']),
                        'novel_terms': '|'.join(result['novel_terms']),
                        
                        # Classification results
                        'classification_label': result['classification_label'],
                        'classification_confidence': result['classification_confidence'],
                        'reasoning': result['reasoning'],
                        'evidence_spans_count': len(result['evidence_spans']),
                        'evidence_spans': '|'.join(result['evidence_spans']),
                        
                        # Rubric assessment (NEW)
                        'rubric_level_achieved': result['rubric_level_achieved'],
                        'rubric_level_100': result['rubric_level_100'],
                        'rubric_level_50': result['rubric_level_50'],
                        'rubric_level_0': result['rubric_level_0'],
                        
                        # Novel term flagging (NEW)
                        'flagged_novel_terms_count': result['flagged_novel_terms_count'],
                        'high_priority_terms': '|'.join(result['high_priority_terms']),
                        'medium_priority_terms': '|'.join(result['medium_priority_terms']),
                        
                        # Latent signals (NEW)
                        'latent_mechanism_explanations': '|'.join(result['latent_mechanism_explanations']),
                        'latent_novel_terms_in_mechanisms': '|'.join(result['latent_novel_terms_in_mechanisms']),
                        'latent_critical_engagement': result['latent_critical_engagement'],
                        
                        # Aggregator routing (UPDATED)
                        'aggregator_recommendation': 'ROUTE' if result['aggregator_recommendation'] else 'BASELINE',
                        'aggregator_reason': result['aggregator_reason'],
                        
                        # Error tracking
                        'error': result['error'] or ''
                    }
                    
                    writer.writerow(row)
            
            logger.info(f"Results saved to {output_path}")
            console.print(f"\n[green]✓ Results saved to: {output_path}[/green]")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            console.print(f"[red]✗ Failed to save results: {e}[/red]")
            return None
    
    def print_model_info(self):
        """Print information about configured LLM models."""
        try:
            extractor_info = provider_manager.get_model_info("extractor")
            classifier_info = provider_manager.get_model_info("classifier")
            
            info_text = (
                f"[bold]Extractor:[/bold] {extractor_info['provider']} - {extractor_info['model']}\n"
                f"[bold]Classifier:[/bold] {classifier_info['provider']} - {classifier_info['model']}\n"
                f"[bold]Temperature:[/bold] Extractor={extractor_info['temperature']}, "
                f"Classifier={classifier_info['temperature']}"
            )
            
            info_panel = Panel(
                info_text,
                title="Agent Configuration"
            )
            
            console.print(info_panel)
        except Exception as e:
            console.print(f"[red]Failed to get model info: {e}[/red]")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test Classifier Agent on student answers with full pipeline (Extractor → Classifier)"
    )
    
    parser.add_argument(
        "--question-id",
        type=int,
        required=True,
        help="Question ID to test (1-4)"
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
        help="Comma-separated student IDs (e.g., S0001,S0002) or range (e.g., S0001-S0010)"
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
    
    parser.add_argument(
        "--extractor-csv",
        type=str,
        default=None,
        help="Path to extractor results CSV file (skip extraction step if provided)"
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
            # Comma-separated format
            student_ids = [s.strip() for s in args.students.split(',')]
        
        console.print(f"[yellow]Testing specific students: {', '.join(student_ids)}[/yellow]")
    
    # Get database URL
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        console.print("[red]Error: DATABASE_URL not set[/red]")
        return
    
    # Initialize tester
    tester = ClassifierTester(db_url)
    
    try:
        # Connect to database
        await tester.connect_db()
        
        # Load extractor results from CSV if provided
        if args.extractor_csv:
            console.print(f"\n[cyan]Loading extractor results from: {args.extractor_csv}[/cyan]")
            tester.load_extractor_results_from_csv(args.extractor_csv)
        
        # Display configuration
        console.print("[bold cyan]=== CLASSIFIER AGENT TEST ===[/bold cyan]")
        tester.print_model_info()
        
        if args.extractor_csv:
            console.print(f"[yellow]Mode: Using cached extractor results from CSV[/yellow]")
        else:
            console.print(f"[yellow]Mode: Running full pipeline (Extractor → Classifier)[/yellow]")
        
        # Run tests
        console.print(
            f"\n[bold]Testing Classifier on Question {args.question_id}[/bold]"
        )
        
        results = await tester.test_batch(args.question_id, args.limit, student_ids)
        
        if results:
            # Display results
            tester.display_results()
            
            # Save results
            output_file = tester.save_results(args.output)
            
            console.print("\n[bold green]✓ Test completed successfully![/bold green]")
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
