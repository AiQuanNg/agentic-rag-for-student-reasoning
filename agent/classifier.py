"""
Classifier Agent - Research-Focused Classification for Student Reasoning Discovery.

Classifies student answers into three categories:
- STANDARD: Comprehension & Recall (surface-level)
- LATENT: Analysis & Synthesis & Evaluation (deeper reasoning) - PRIMARY RESEARCH FOCUS
- OFF_TOPIC: No relevant understanding

Classification Flow:
1. Get question context (question_text with sub-questions)
2. Call get_question_rubrics() - Get 0/50/100 level rubrics
3. Call get_classification_criteria() - Get generic STANDARD/LATENT definitions
4. Filter OFF_TOPIC (comprehension check)
5. Classify STANDARD vs LATENT (rubric-based depth check + latent signals)
6. Flag novel terms with importance scores for Aggregator routing

Uses hybrid approach:
- Leverages Extractor output (topics, novel terms, themes, keywords)
- References rubrics (question-specific) and criteria (generic) from database
- Routes medium-confidence LATENT answers to Aggregator for theme discovery

Architecture: per-agent LLM config, question context integration,
single-attempt strategy (Orchestrator handles retries).
"""

import sys
from pathlib import Path

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import logging
import asyncpg
import json
from typing import Dict, Any
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from agent.models.classification import ClassificationResult
from agent.prompts.classifier import CLASSIFIER_SYSTEM_PROMPT
from agent.tools.rubric import RubricTools
from agent.config.providers import provider_manager

logger = logging.getLogger(__name__)

# Log which model is configured for this agent
model_info = provider_manager.get_model_info("classifier")
logger.info(
    f"Classifier Agent: {model_info['provider'].upper()} - {model_info['model']} "
    f"(temperature={model_info['temperature']})"
)


@dataclass
class ClassifierDependencies:
    """
    Dependencies for Classifier Agent.
    
    Provides database access and question context for tools.
    Similar to ExtractorDependencies but includes question metadata.
    """
    db_pool: asyncpg.Pool
    question_id: int
    question_text: str  # Full question text - LLM infers goal/topic from this


# Initialize Classifier agent with configured provider
classifier_agent = Agent(
    model=provider_manager.get_model("classifier"),  # ← Per-agent model from .env
    deps_type=ClassifierDependencies,
    system_prompt=CLASSIFIER_SYSTEM_PROMPT,
    result_type=ClassificationResult
)


@classifier_agent.tool
async def get_question_rubrics(
    ctx: RunContext[ClassifierDependencies]
) -> Dict[str, Any]:
    """
    Get question-specific rubrics for ALL three levels (0%, 50%, 100%).
    
    Returns rubrics that define:
    - Level 0: Minimal/no understanding
    - Level 50: Partial understanding (definitions, basic concepts)
    - Level 100: Full understanding (mechanisms, implications, connections)
    
    These are question-specific and should be combined with question_text
    to assess answer depth.
    
    Returns:
        {
            "level_0": "Criteria for minimal understanding",
            "level_50": "Criteria for partial understanding", 
            "level_100": "Criteria for full understanding",
            "question_id": int
        }
    """
    try:
        tools = RubricTools(ctx.deps.db_pool)
        rubrics = await tools.get_question_rubrics(ctx.deps.question_id)
        logger.debug(f"Retrieved rubrics for Q{ctx.deps.question_id}: {list(rubrics.keys())}")
        return rubrics
    
    except Exception as e:
        logger.error(f"Failed to get rubrics for Q{ctx.deps.question_id}: {e}")
        return {}  # Return empty dict on error


@classifier_agent.tool
async def get_classification_criteria(
    ctx: RunContext[ClassifierDependencies]
) -> Dict[str, Dict[str, str]]:
    """
    Get generic classification criteria for STANDARD vs LATENT.
    
    Returns definitions that help distinguish between:
    - STANDARD: Comprehension & Recall (surface-level)
    - LATENT: Analysis & Synthesis & Evaluation (deeper reasoning)
    - OFF_TOPIC: No relevant understanding
    
    These are generic guidelines - use question-specific rubrics
    (from get_question_rubrics) for primary depth assessment.
    
    Returns:
        {
            "standard": {
                "definition": "...",
                "indicators": [...]
            },
            "latent": {
                "definition": "...",
                "indicators": [...]
            },
            "off_topic": {
                "definition": "...",
                "indicators": [...]
            }
        }
    """
    try:
        tools = RubricTools(ctx.deps.db_pool)
        criteria = await tools.get_classification_criteria()
        logger.debug(f"Retrieved {len(criteria)} classification criteria categories")
        return criteria
    
    except Exception as e:
        logger.error(f"Failed to get classification criteria: {e}")
        return {}


async def fetch_question_context(
    db_pool: asyncpg.Pool,
    question_id: int
) -> str:
    """
    Fetch question text from database.
    
    Returns:
        question_text: Full question text with sub-questions
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT question_text
                FROM questions
                WHERE id = $1
            """, question_id)
            
            if row:
                return row['question_text']
            else:
                logger.warning(f"Question {question_id} not found in database")
                return ""
    
    except Exception as e:
        logger.error(f"Failed to fetch question text for Q{question_id}: {e}")
        return ""


async def classify_answer(
    db_pool: asyncpg.Pool,
    question_id: int,
    answer_text: str,
    extraction_results: Any,  # ExtractionResult from Extractor
    question_text: str = None
) -> ClassificationResult:
    """
    Classify student answer as Standard, Latent, or Off-topic.
    
    Classification Flow:
    1. Get question context (fetch question_text if not provided)
    2. Call get_question_rubrics() - 0/50/100 level rubrics
    3. Call get_classification_criteria() - Generic STANDARD/LATENT definitions
    4. Filter OFF_TOPIC (comprehension check)
    5. Classify STANDARD vs LATENT (rubric-based + latent signals)
    6. Flag novel terms with importance scores
    
    Uses hybrid approach:
    1. Student answer text
    2. Question text (LLM infers goal/topic/sub-questions from this)
    3. Extraction results (topics, novel terms, themes, keywords)
    4. Rubrics (question-specific 0/50/100 levels)
    5. Criteria (generic STANDARD/LATENT definitions)
    
    Args:
        db_pool: Database connection pool
        question_id: Which question (1-4)
        answer_text: Student's full answer
        extraction_results: Output from Extractor Agent (ExtractionResult)
        question_text: Optional - full question text (fetched if not provided)
    
    Returns:
        ClassificationResult with:
        - label (standard|latent|off_topic)
        - classification_confidence
        - rubric_assessment (Level 0/50/100 alignment)
        - flagged_novel_terms (with importance scores for Aggregator)
        - latent_signals_summary
        - evidence_spans
        - reasoning
        - aggregator_recommendation
    """
    try:
        # Step 1: Get question text if not provided
        if not question_text:
            logger.info(f"Fetching question text for Q{question_id}")
            question_text = await fetch_question_context(db_pool, question_id)
        
        # Create dependencies with question context
        deps = ClassifierDependencies(
            db_pool=db_pool,
            question_id=question_id,
            question_text=question_text
        )
        
        # Build classification prompt following the flow
        classification_prompt = f"""Classify this student answer following the three-step process.

## QUESTION CONTEXT (Step 1 - Understand the Question)
Question #{question_id}:
{question_text}

**Your task**: Identify the question goal (e.g., "Learn basics", "Apply knowledge", "Evaluate risks"), topic area (e.g., "fundamentals", "applications", "ethics"), and sub-questions from the question text above.

## STUDENT ANSWER
{answer_text}

## EXTRACTOR FINDINGS (Three-Tier Output)

**Tier 1: Topics** (Sub-questions addressed)
{', '.join(extraction_results.topic) if extraction_results.topic else 'None'}

**Tier 2: Novel Terms** (New concepts - PRIMARY FOCUS)
{', '.join(extraction_results.novel_terms) if extraction_results.novel_terms else 'None'}

**Tier 3: Themes** (Broad categories)
{', '.join(extraction_results.detected_themes) if extraction_results.detected_themes else 'None'}

**Legacy Fields:**
- Matched Keywords: {', '.join(extraction_results.matched_keywords) if extraction_results.matched_keywords else 'None'}
- Extraction Confidence: {extraction_results.extraction_confidence:.2f}

## CLASSIFICATION TASK

Follow this exact flow:

**Step 2: Get Tools (REQUIRED)**
1. Call get_question_rubrics() → Get 0/50/100 level rubrics for Q{question_id}
2. Call get_classification_criteria() → Get generic STANDARD/LATENT definitions

**Step 3: Comprehension Check (Off-topic Filter)**
- Does answer meet Level 0 rubric?
- Are there matched_keywords OR novel_terms?
- If NO → OFF_TOPIC

**Step 4: Rubric-Based Depth Assessment**
- Does answer meet Level 100 rubric? (REQUIRED for LATENT)
  - YES → Check latent signals (mechanisms, novel terms, critical thinking)
  - NO → Check Level 50
    - Meets Level 50 → STANDARD
    - Below Level 50 → OFF_TOPIC or LOW STANDARD

**Step 5: Novel Term Flagging**
- For each novel term from Tier 2, calculate importance score:
  - Specificity (40%): Multi-word technical compound vs. generic
  - Usage depth (30%): In mechanism explanation vs. just listed
  - Rubric alignment (20%): Contributes to Level 100 understanding
  - Frequency (10%): How often term appears
- Flag HIGH/MEDIUM/LOW priority for Aggregator routing

## OUTPUT REQUIREMENTS

Return JSON matching ClassificationResult schema with:
- label: "standard"|"latent"|"off_topic"
- classification_confidence: 0.0-1.0
- rubric_assessment: Level 0/50/100 met, evidence quotes
- flagged_novel_terms: Each term with importance_score, priority, evidence_spans
- latent_signals_summary: Mechanism quotes, critical engagement, etc.
- three_tier_context: Topics, novel terms, themes counts
- question_alignment: Include inferred goal and topic from question_text
- aggregator_recommendation: ROUTE|STORE|BASELINE with reason

**CRITICAL**: LATENT classification REQUIRES Level 100 rubric met. If Level 100 not met, classify as STANDARD or OFF_TOPIC."""
        
        # Get temperature from config
        config = provider_manager.get_agent_config("classifier")
        
        # Run classifier (single attempt)
        logger.info(f"Running classifier for Q{question_id}...")
        result = await classifier_agent.run(
            classification_prompt,
            deps=deps,
            model_settings={"temperature": config.temperature}
        )
        
        logger.info(
            f"Classification complete: Q{question_id}, "
            f"label={result.data.label}, "
            f"confidence={result.data.classification_confidence:.2f}, "
            f"rubric_level={result.data.rubric_assessment.get('rubric_level_achieved', 'unknown') if hasattr(result.data, 'rubric_assessment') and result.data.rubric_assessment else 'unknown'}"
        )
        
        # Log novel terms flagged
        if hasattr(result.data, 'flagged_novel_terms') and result.data.flagged_novel_terms:
            high_priority = sum(1 for t in result.data.flagged_novel_terms if t.get('priority') == 'HIGH')
            logger.info(f"Flagged {len(result.data.flagged_novel_terms)} novel terms ({high_priority} HIGH priority)")
        
        return result.data
    
    except Exception as e:
        logger.error(f"Classification failed for Q{question_id}: {e}", exc_info=True)
        
        # Return minimal result on failure
        return ClassificationResult(
            label="off_topic",
            classification_confidence=0.0,
            evidence_spans=[],
            reasoning=f"Classification failed: {str(e)}",
            rubric_alignment={
                "matches_level_100": False,
                "matches_level_50": False,
                "matches_level_0": False,
                "rubric_reasoning": "Classification error"
            },
            extractor_context={
                "topics_found": len(extraction_results.topic) if extraction_results and extraction_results.topic else 0,
                "novel_terms_found": len(extraction_results.novel_terms) if extraction_results and extraction_results.novel_terms else 0,
                "keywords_found": len(extraction_results.matched_keywords) if extraction_results and extraction_results.matched_keywords else 0,
                "extraction_confidence": extraction_results.extraction_confidence if extraction_results else 0.0
            },
            criteria_assessment={
                "error": [str(e)]  # Changed to list to match Dict[str, List[str]]
            },
            flagged_novel_terms=[],
            aggregator_recommendation={
                "route_to_aggregator": False,
                "reason": "Classification error"
            }
        )