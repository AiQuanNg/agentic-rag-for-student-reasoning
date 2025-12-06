"""
Extractor Agent with Per-Agent LLM Configuration.
"""

import sys
from pathlib import Path

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import logging
import asyncpg
from typing import List
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

from agent.models.extraction import ExtractionResult
from agent.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT
from agent.tools.codebook import CodebookTools
from agent.config.providers import provider_manager

logger = logging.getLogger(__name__)

# Log which model is configured for this agent
model_info = provider_manager.get_model_info("extractor")
logger.info(
    f"Extractor Agent: {model_info['provider'].upper()} - {model_info['model']} "
    f"(temperature={model_info['temperature']})"
)


@dataclass
class ExtractorDependencies:
    """Dependencies for Extractor Agent."""
    db_pool: asyncpg.Pool
    question_id: int


# Initialize Extractor agent with configured provider
extractor_agent = Agent(
    model=provider_manager.get_model("extractor"),  # ← Per-agent model from .env
    deps_type=ExtractorDependencies,
    system_prompt=EXTRACTOR_SYSTEM_PROMPT,
    result_type=ExtractionResult
)


@extractor_agent.tool
async def retrieve_codebook_keywords(
    ctx: RunContext[ExtractorDependencies]
) -> List[str]:
    """Retrieve approved keywords for extraction."""
    try:
        tools = CodebookTools(ctx.deps.db_pool)
        keywords = await tools.get_approved_keywords(ctx.deps.question_id)
        logger.debug(f"Retrieved {len(keywords)} keywords")
        return keywords
    except Exception as e:
        logger.error(f"Keyword retrieval failed: {e}")
        return []


async def extract_keywords(
    db_pool: asyncpg.Pool,
    question_id: int,
    answer_text: str
) -> ExtractionResult:
    """Extract keywords from student answer using configured Extractor model."""
    try:
        deps = ExtractorDependencies(db_pool=db_pool, question_id=question_id)
        
        prompt = f"""Extract keywords from student answer (Q{question_id}):

{answer_text}

Use retrieve_codebook_keywords to see approved vocabulary.
Output ONLY JSON."""
        
        # Get temperature from config
        config = provider_manager.get_agent_config("extractor")
        
        result = await extractor_agent.run(
            prompt,
            deps=deps,
            model_settings={"temperature": config.temperature}
        )
        
        logger.info(f"Extraction complete: confidence={result.data.extraction_confidence}")
        return result.data
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return ExtractionResult(
            topic=[], # New test field
            matched_keywords=[],
            detected_themes=[],
            novel_terms=[],
            evidence_spans=[],
            extraction_confidence=0.0
        )
