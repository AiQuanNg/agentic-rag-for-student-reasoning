"""
Classifier Agent - Two-Layer Classification Strategy for Student Reasoning Discovery.

Classifies student answers into three categories:
- STANDARD: Meets rubric expectations (Level 50 or Level 100 without latent signals)
- LATENT: Meets Level 100 + demonstrates latent reasoning beyond rubric (PRIMARY RESEARCH FOCUS)
- OFF_TOPIC: No relevant understanding

Classification Flow (Three Layers):
1. LAYER 1: Pure rubric grading (categorical: 0/50/100)
   - Assess answer against question-specific rubrics ONLY
   - Output: level_achieved (0, 50, or 100)
   - Level 100 = eligible for Layer 2

2. LAYER 2: Latent signal detection (only if Level 100)
   - Start from 0.0, score pure latent signals
   - Components: mechanisms (0.30), novel terms (0.10), critical engagement (0.30),
                 evidence (0.20), cross-domain (0.10)
   - Output: latent_score (0.0-1.0)
   - Score >= 0.75 → LATENT (high), >= 0.60 → LATENT (medium), < 0.60 → STANDARD

3. LAYER 3: Novel term flagging (skip for OFF_TOPIC)
   - Calculate importance scores for all novel terms
   - Flag HIGH/MEDIUM/LOW priority for Aggregator routing

Uses hybrid approach:
- Leverages Extractor output (topics, novel terms, themes, keywords)
- References rubrics (question-specific) and criteria (generic) from database
- Routes medium-confidence LATENT answers to Aggregator for theme discovery

Architecture: per-agent LLM config, question context integration,
single-attempt strategy (Orchestrator handles retries).

Version: 4.0 - Two-Layer Separation: Rubric Grading → Latent Detection
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
    Classify student answer as Standard, Latent, or Off-topic using two-layer approach.
    
    Classification Flow (Three Layers):
    1. LAYER 1: Pure rubric grading (categorical: 0/50/100)
       - Assess against question-specific rubrics only
       - No latent signal consideration
    
    2. LAYER 2: Latent signal detection (only if Level 100)
       - Start from 0.0, score pure latent signals
       - Mechanisms (0.30), novel terms (0.10), critical (0.30), evidence (0.20), cross-domain (0.10)
       - >= 0.75 → LATENT (high), >= 0.60 → LATENT (medium), < 0.60 → STANDARD
    
    3. LAYER 3: Novel term flagging (skip for OFF_TOPIC)
       - Calculate importance scores for all novel terms
       - Flag HIGH/MEDIUM/LOW priority for Aggregator
    
    Args:
        db_pool: Database connection pool
        question_id: Which question (1-4)
        answer_text: Student's full answer
        extraction_results: Output from Extractor Agent (ExtractionResult)
        question_text: Optional - full question text (fetched if not provided)
    
    Returns:
        ClassificationResult with:
        - layer_1_rubric_grading: Level achieved (0/50/100), evidence, reasoning
        - layer_2_latent_detection: Signal scores, total_latent_score (only if level=100)
        - layer_3_novel_terms: All novel terms with importance scores
        - classification_reasoning: Summary of all three layers
        - Legacy fields for backward compatibility (rubric_assessment, latent_signals_summary, flagged_novel_terms)
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
        
        # Build classification prompt following the three-layer flow
        classification_prompt = f"""Classify this student answer following the three-layer process.

## QUESTION CONTEXT
Question #{question_id}:
{question_text}

**Your task**: Understand the question goal, topic area, and sub-questions from the question text above.

## STUDENT ANSWER
{answer_text}

## EXTRACTOR FINDINGS (Three-Tier Output)

**Tier 1: Topics** (Sub-questions addressed)
{', '.join(extraction_results.topic) if extraction_results.topic else 'None'}

**Tier 2: Novel Terms** (New concepts - PRIMARY FOCUS)
Novel terms found: {', '.join(extraction_results.novel_terms) if extraction_results.novel_terms else 'None'}
**Count: {len(extraction_results.novel_terms) if extraction_results.novel_terms else 0} terms**

**Tier 3: Themes** (Broad categories)
{', '.join(extraction_results.detected_themes) if extraction_results.detected_themes else 'None'}

**Legacy Fields:**
- Matched Keywords: {', '.join(extraction_results.matched_keywords) if extraction_results.matched_keywords else 'None'}
- Extraction Confidence: {extraction_results.extraction_confidence:.2f}

## CLASSIFICATION TASK - THREE-LAYER PROCESS

**STEP 1: Get Tools (REQUIRED)**
1. Call get_question_rubrics() → Get Level 0/50/100 rubrics for Q{question_id}
2. Call get_classification_criteria() → Get generic STANDARD/LATENT definitions

**STEP 2: LAYER 1 - Pure Rubric Grading (Categorical: 0/50/100)**
- Compare answer against Level 100, then Level 50, then Level 0 rubrics
- Determine which level is achieved (categorical, not numeric)
- Output: level_achieved (0, 50, or 100)
- **CRITICAL**: This is pure rubric alignment - do NOT consider latent signals here

**STEP 3: LAYER 2 - Latent Signal Detection (ONLY if Level 100)**
- **Eligibility check**: Only run if level_achieved = 100
- **Start from 0.0** - Calculate pure latent score from scratch:
  - Mechanism explanations: 0.0 - 0.30 (beyond rubric expectations)
  - Novel terms in context: 0.0 - 0.10 (well-integrated usage)
  - Critical engagement: 0.0 - 0.30 (trade-offs, analysis)
  - Evidence & specificity: 0.0 - 0.20 (research quality)
  - Cross-domain thinking: 0.0 - 0.10 (synthesis)
- **Total latent_score**: 0.0 - 1.00 (sum of above, NO modulation)
- **Classification**:
  - latent_score >= 0.75 → LATENT (high confidence)
  - latent_score >= 0.60 → LATENT (medium confidence, flag for Aggregator)
  - latent_score < 0.60 → STANDARD (meets 100 but not latent)
- If Level 100 NOT achieved:
  - Level 50 → STANDARD
  - Level 0 → OFF_TOPIC

**STEP 4: LAYER 3 - Novel Term Flagging (Skip for OFF_TOPIC)**
- **You MUST flag all {len(extraction_results.novel_terms) if extraction_results.novel_terms else 0} novel terms from Tier 2**
- Calculate importance scores using the formula from system prompt
- Assign priority: HIGH (≥0.70) / MEDIUM (≥0.50) / LOW (<0.50)
- Include evidence_spans and usage_context for each term
- **Skip if classification = OFF_TOPIC**

## OUTPUT REQUIREMENTS

Return JSON matching ClassificationResult schema with:

**New Structure (v4.0)**:
- layer_1_rubric_grading: Level achieved (0/50/100), evidence, reasoning
- layer_2_latent_detection: Signal scores, total_latent_score, classification (only if level=100)
- layer_3_novel_terms: All novel terms with importance scores (empty for OFF_TOPIC)
- classification_reasoning: Summary of all three layers

**Legacy Fields (backward compatibility)**:
- rubric_assessment: Maps to layer_1_rubric_grading
- latent_signals_summary: Maps to layer_2_latent_detection.signal_evidence
- flagged_novel_terms: Maps to layer_3_novel_terms

**CRITICAL RULES**:
1. **Two-layer separation**: Layer 1 (rubric) is INDEPENDENT of Layer 2 (latent)
2. **No undergrading**: Level 100 with low latent score → STANDARD (not Level 50)
3. **LATENT requires BOTH**: Level 100 (Layer 1) AND latent_score >= 0.60 (Layer 2)
4. **Layer 2 starts from 0.0**: No base score, pure latent signal scoring
5. **OFF_TOPIC skips Layer 3**: Novel term flagging only for STANDARD/LATENT
6. **Flag ALL novel terms**: Even STANDARD answers may have valuable terms
"""
        
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
            f"layer_1_level={result.data.layer_1_rubric_grading.get('level_achieved', 'unknown') if hasattr(result.data, 'layer_1_rubric_grading') and result.data.layer_1_rubric_grading else 'unknown'}, "
            f"layer_2_score={result.data.layer_2_latent_detection.get('total_latent_score', 'N/A') if hasattr(result.data, 'layer_2_latent_detection') and result.data.layer_2_latent_detection else 'N/A'}"
        )
        
        # Log novel terms flagged (Layer 3)
        if hasattr(result.data, 'layer_3_novel_terms') and result.data.layer_3_novel_terms:
            high_priority = sum(1 for t in result.data.layer_3_novel_terms if t.get('priority') == 'HIGH')
            logger.info(f"Layer 3: Flagged {len(result.data.layer_3_novel_terms)} novel terms ({high_priority} HIGH priority)")
        elif hasattr(result.data, 'flagged_novel_terms') and result.data.flagged_novel_terms:
            # Fallback to legacy field
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
            layer_1_rubric_grading={
                "level_achieved": 0,
                "rubric_evidence": [],
                "latent_eligible": False,
                "grading_reasoning": "Classification error"
            },
            layer_2_latent_detection=None,
            layer_3_novel_terms=[],
            classification_reasoning={
                "layer_1_summary": "Classification failed",
                "layer_2_summary": "Skipped due to error",
                "layer_3_summary": "Skipped due to error",
                "final_decision": f"Classification error: {str(e)}"
            },
            rubric_assessment={
                "rubric_level_achieved": "0",
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