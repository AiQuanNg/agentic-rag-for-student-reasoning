"""
Classifier Agent - Research-Focused Classification for Student Reasoning Discovery.

Classifies student answers into three categories:
- STANDARD: Comprehension & Recall (surface-level)
- LATENT: Analysis & Synthesis & Evaluation (deeper reasoning) - PRIMARY RESEARCH FOCUS
- OFF_TOPIC: No relevant understanding

Uses hybrid approach:
- Leverages Extractor output (keywords, themes, novel terms)
- References rubrics and criteria from database
- Routes low-confidence LATENT answers to Aggregator for theme discovery

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
    
    # Optional: Question context (fetched upstream, not via tool)
    question_text: str = None
    question_goal: str = None
    question_topic: str = None


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
) -> Dict[str, str]:
    """
    Get ALL rubrics for this question (no filtering).
    
    Returns rubrics at all three levels (100%, 50%, 0%) as context
    for LLM classification. NOT similarity-based retrieval.
    """
    try:
        tools = RubricTools(ctx.deps.db_pool)
        rubrics = await tools.get_question_rubrics(ctx.deps.question_id)
        logger.debug(f"Retrieved {len(rubrics)} rubrics for Q{ctx.deps.question_id}")
        return rubrics
    
    except Exception as e:
        logger.error(f"Failed to get rubrics for Q{ctx.deps.question_id}: {e}")
        return {}  # Return empty dict on error


@classifier_agent.tool
async def get_classification_criteria(
    ctx: RunContext[ClassifierDependencies]
) -> Dict[str, Dict[str, str]]:
    """
    Get Standard vs Latent classification criteria.
    
    Returns definitions that help distinguish between:
    - Standard answers (Comprehension & Recall)
    - Latent answers (Analysis & Synthesis & Evaluation)
    """
    try:
        tools = RubricTools(ctx.deps.db_pool)
        criteria = await tools.get_classification_criteria()
        logger.debug(f"Retrieved {len(criteria)} classification criteria")
        return criteria
    
    except Exception as e:
        logger.error(f"Failed to get classification criteria: {e}")
        return {}


async def classify_answer(
    db_pool: asyncpg.Pool,
    question_id: int,
    answer_text: str,
    extraction_results: Any,  # ExtractionResult from Extractor
    question_text: str = None,
    question_goal: str = None,
    question_topic: str = None
) -> ClassificationResult:
    """
    Classify student answer as Standard, Latent, or Off-topic.
    
    Uses hybrid approach:
    1. Student answer text
    2. Question context (optional but recommended)
    3. Extraction results (keywords, themes, novel terms)
    4. Rubrics and criteria from database tools
    
    Single-attempt strategy - Orchestrator manages retries.
    
    Args:
        db_pool: Database connection pool
        question_id: Which question (1-4)
        answer_text: Student's full answer
        extraction_results: Output from Extractor Agent
        question_text: Optional - exact question text
        question_goal: Optional - learning goal (e.g., "Learn basics")
        question_topic: Optional - topic area (e.g., "fundamentals")
    
    Returns:
        ClassificationResult with label, confidence, reasoning, evidence, etc.
    """
    try:
        # Create dependencies with question context
        deps = ClassifierDependencies(
            db_pool=db_pool,
            question_id=question_id,
            question_text=question_text,
            question_goal=question_goal,
            question_topic=question_topic
        )
        
        # Build classification prompt with all context
        classification_prompt = f"""Classify this student answer.

QUESTION CONTEXT:
Question #{question_id}
{f"Text: {question_text}" if question_text else ""}
{f"Goal: {question_goal}" if question_goal else ""}
{f"Topic: {question_topic}" if question_topic else ""}

STUDENT ANSWER:
{answer_text}

EXTRACTOR FINDINGS:
- Keywords: {', '.join(extraction_results.matched_keywords) if extraction_results.matched_keywords else 'None'}
- Themes: {', '.join(extraction_results.detected_themes) if extraction_results.detected_themes else 'None'}
- Novel Terms: {', '.join(extraction_results.novel_terms) if extraction_results.novel_terms else 'None'}
- Confidence: {extraction_results.extraction_confidence:.2f}

CLASSIFICATION TASK:
1. Call get_question_rubrics() to see rubric expectations
2. Call get_classification_criteria() to understand Standard vs Latent
3. Analyze for LATENT signals (primary research focus)
4. Determine if STANDARD, LATENT, or OFF_TOPIC

Output ONLY JSON matching ClassificationResult schema."""
        
        # Get temperature from config
        config = provider_manager.get_agent_config("classifier")
        
        # Run classifier (single attempt)
        result = await classifier_agent.run(
            classification_prompt,
            deps=deps,
            model_settings={"temperature": config.temperature}
        )
        
        logger.info(
            f"Classification complete: Q{question_id}, "
            f"label={result.data.label}, "
            f"confidence={result.data.classification_confidence:.2f}"
        )
        
        return result.data
    
    except Exception as e:
        logger.error(f"Classification failed for Q{question_id}: {e}")
        
        # Return minimal result on failure
        return ClassificationResult(
            label="off_topic",
            classification_confidence=0.0,
            evidence_spans=[],
            reasoning=f"Classification failed: {str(e)}",
            rubric_alignment={},
            extractor_context={},
            criteria_assessment={}
        )
