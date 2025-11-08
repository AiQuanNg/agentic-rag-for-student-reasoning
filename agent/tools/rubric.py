"""
Tools for accessing rubrics and classification criteria in the Classifier Agent.

Provides retrieval of:
- Question-specific rubrics (all levels: 0%, 50%, 100%)
- Classification criteria (Standard vs Latent definitions)

Adapted from ottomator-agents/tools.py pattern and codebook.py structure.
Uses simple database fetch (NOT pgvector similarity) for hybrid classification.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncpg

logger = logging.getLogger(__name__)


class RubricTools:
    """
    Tools for retrieving rubrics and criteria for the Classifier Agent.
    
    Unlike pgvector similarity search, this uses simple database queries
    to fetch ALL rubrics for a question, allowing LLM to reason about
    alignment rather than relying on low-similarity scores.
    
    Follows CodebookTools pattern for consistency across agents.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize rubric tools with database connection pool.
        
        Args:
            db_pool: AsyncPG connection pool for database access
        """
        self.pool = db_pool
    
    async def get_question_rubrics(self, question_id: int) -> Dict[str, str]:
        """
        Get ALL rubrics for a specific question (no filtering).
        
        Returns rubrics at all three levels (100%, 50%, 0%) as context
        for LLM classification. This is NOT similarity-based retrieval.
        
        Primary tool for Classifier Agent - LLM uses these as reference
        to determine which level the student answer aligns with.
        
        Args:
            question_id: Question ID to fetch rubrics for
            
        Returns:
            Dictionary with rubric levels as keys:
            {
                'level_100': 'Full understanding: Answer is 200-300 words...',
                'level_50': 'Partial understanding: Answer provided but...',
                'level_0': 'No understanding: Answer missing or...'
            }
            
        Example:
            >>> tools = RubricTools(db_pool)
            >>> rubrics = await tools.get_question_rubrics(question_id=1)
            >>> print(rubrics['level_100'])
            'Full understanding: Answer is 200-300 words, uses clear logic...'
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT level_pct, descriptor
                    FROM rubrics
                    WHERE question_id = $1
                    ORDER BY level_pct DESC
                """, question_id)
                
                # Build dictionary with level keys
                rubrics_dict = {}
                for row in rows:
                    level = f"level_{row['level_pct']}"
                    rubrics_dict[level] = row['descriptor']
                
                logger.info(f"Retrieved {len(rubrics_dict)} rubrics for question {question_id}")
                return rubrics_dict
                
        except Exception as e:
            logger.error(f"Failed to retrieve rubrics for question {question_id}: {e}")
            return {}  # Return empty dict on error
    
    async def get_classification_criteria(self) -> Dict[str, Dict[str, str]]:
        """
        Get Standard vs Latent classification criteria.
        
        Returns definitions that help distinguish between:
        - Standard answers (use expected terminology, surface-level)
        - Latent answers (demonstrate deeper reasoning, non-standard approach)
        
        This is a simple lookup - no question_id filtering needed since
        criteria apply across all questions.
        
        Returns:
            Dictionary with criteria names as keys:
            {
                'Standard Answer': {
                    'name': 'Standard Answer',
                    'description': 'Meets rubric fully with accurate...',
                    'guidance': 'Comprehension & Recall: The answer...'
                },
                'Latent Answer': {
                    'name': 'Latent Answer',
                    'description': 'Meets rubric fully with deeper...',
                    'guidance': 'Analysis, Synthesis, or Evaluation...'
                }
            }
            
        Example:
            >>> tools = RubricTools(db_pool)
            >>> criteria = await tools.get_classification_criteria()
            >>> print(criteria['Standard Answer']['description'])
            'Meets rubric fully with accurate, clear but surface-level reasoning'
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT name, description, guidance
                    FROM criteria
                    WHERE name IN ('Standard Answer', 'Latent Answer')
                """)
                
                # Build nested dictionary
                criteria_dict = {}
                for row in rows:
                    criteria_dict[row['name']] = {
                        'name': row['name'],
                        'description': row['description'],
                        'guidance': row['guidance']
                    }
                
                logger.info(f"Retrieved {len(criteria_dict)} classification criteria")
                return criteria_dict
                
        except Exception as e:
            logger.error(f"Failed to retrieve classification criteria: {e}")
            return {}
    
    async def get_rubrics_with_metadata(self, question_id: int) -> List[Dict[str, Any]]:
        """
        Get rubrics with full metadata for debugging/inspection.
        
        Similar to get_keywords_with_metadata() in codebook.py.
        Useful for testing and validation, not typically used by agent.
        
        Args:
            question_id: Question ID to fetch rubrics for
            
        Returns:
            List of rubric dictionaries with all columns
            
        Example:
            >>> tools = RubricTools(db_pool)
            >>> rubrics = await tools.get_rubrics_with_metadata(1)
            >>> print(rubrics[0])
            {'id': 1, 'question_id': 1, 'level_pct': 100, 'descriptor': '...', 'exemplar': '...'}
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id,
                        question_id,
                        level_pct,
                        descriptor,
                        exemplar
                    FROM rubrics
                    WHERE question_id = $1
                    ORDER BY level_pct DESC
                """, question_id)
                
                return [
                    {
                        'id': row['id'],
                        'question_id': row['question_id'],
                        'level_pct': row['level_pct'],
                        'descriptor': row['descriptor'],
                        'exemplar': row['exemplar']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to retrieve rubric metadata for question {question_id}: {e}")
            return []
    
    async def count_rubrics(self, question_id: int) -> int:
        """
        Count rubrics for a question (should always be 3: levels 0, 50, 100).
        
        Useful for validation - ensure question has complete rubric set.
        Similar to count_keywords() in codebook.py.
        
        Args:
            question_id: Question ID
            
        Returns:
            Count of rubrics (expected: 3)
            
        Example:
            >>> tools = RubricTools(db_pool)
            >>> count = await tools.count_rubrics(question_id=1)
            >>> assert count == 3, "Question missing rubric levels!"
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT COUNT(*) as count
                    FROM rubrics
                    WHERE question_id = $1
                """, question_id)
                
                return row['count'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to count rubrics for question {question_id}: {e}")
            return 0
