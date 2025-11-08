"""
Tools for accessing the thematic codebook in the Extractor Agent.
Provides retrieval of approved keywords for semantic matching.

Schema: topic_keywords (id, question_id, keyword, status, source)
- status: 'approved' or 'pending'
- source: 'professor' or 'system'
"""

import logging
from typing import List, Dict, Any
import asyncpg

logger = logging.getLogger(__name__)


class CodebookTools:
    """
    Tools for retrieving approved keywords from thematic codebook.
    
    The LLM will:
    1. Receive keyword list from this tool
    2. Semantically match keywords to student answer
    3. Independently detect themes (NOT from database)
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize codebook tools with database connection pool.
        
        Args:
            db_pool: AsyncPG connection pool for database access
        """
        self.pool = db_pool
    
    async def get_approved_keywords(self, question_id: int) -> List[str]:
        """
        Get approved keywords for a specific question.
        
        Primary tool for Extractor Agent - returns simple list of keyword strings.
        LLM will semantically match these to student answer and detect themes itself.
        
        Args:
            question_id: Question ID to filter keywords
            
        Returns:
            List of approved keyword strings, sorted alphabetically
            
        Example:
            >>> tools = CodebookTools(db_pool)
            >>> keywords = await tools.get_approved_keywords(question_id=1)
            >>> print(keywords)
            ['adversarial', 'ai', 'algorithm', 'align', 'alphacode', 'anatomy', ...]
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT keyword
                    FROM topic_keywords
                    WHERE question_id = $1 
                      AND status = 'approved'
                    ORDER BY keyword
                """, question_id)
                
                return [row['keyword'] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to retrieve keywords for question {question_id}: {e}")
            return []
    
    async def get_keywords_with_metadata(self, question_id: int) -> List[Dict[str, Any]]:
        """
        Get approved keywords with metadata (source info).
        
        Useful for debugging - shows whether keyword came from professor or system.
        
        Args:
            question_id: Question ID to filter keywords
            
        Returns:
            List of dicts with keyword, status, source
            
        Example:
            >>> keywords = await tools.get_keywords_with_metadata(1)
            >>> print(keywords[0])
            {'keyword': 'adversarial', 'status': 'approved', 'source': 'professor'}
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        id,
                        keyword,
                        status,
                        source
                    FROM topic_keywords
                    WHERE question_id = $1 
                      AND status = 'approved'
                    ORDER BY keyword
                """, question_id)
                
                return [
                    {
                        'id': row['id'],
                        'keyword': row['keyword'],
                        'status': row['status'],
                        'source': row['source']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to retrieve keyword metadata for question {question_id}: {e}")
            return []
    
    async def get_all_approved_keywords(self) -> List[Dict[str, Any]]:
        """
        Get all approved keywords across all questions.
        
        Useful for:
        - Analytics/reporting
        - Cross-question pattern detection
        - Codebook export
        
        Returns:
            List of dicts with keyword, question_id, source
            
        Example:
            >>> all_keywords = await tools.get_all_approved_keywords()
            >>> print(f"Total approved keywords: {len(all_keywords)}")
            >>> print(all_keywords[0])
            {'keyword': 'adversarial', 'question_id': 1, 'source': 'professor'}
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT 
                        tk.id,
                        tk.keyword,
                        tk.question_id,
                        tk.status,
                        tk.source,
                        q.question_text
                    FROM topic_keywords tk
                    JOIN questions q ON q.id = tk.question_id
                    WHERE tk.status = 'approved'
                    ORDER BY tk.question_id, tk.keyword
                """)
                
                return [
                    {
                        'id': row['id'],
                        'keyword': row['keyword'],
                        'question_id': row['question_id'],
                        'source': row['source'],
                        'question_text': row['question_text']
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Failed to retrieve all approved keywords: {e}")
            return []
    
    async def count_keywords(self, question_id: int) -> int:
        """
        Count approved keywords for a question.
        
        Useful for validation - ensure question has sufficient codebook coverage.
        
        Args:
            question_id: Question ID
            
        Returns:
            Count of approved keywords
            
        Example:
            >>> count = await tools.count_keywords(question_id=1)
            >>> print(f"Question 1 has {count} approved keywords")
            Question 1 has 47 approved keywords
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT COUNT(*) as count
                    FROM topic_keywords
                    WHERE question_id = $1 
                      AND status = 'approved'
                """, question_id)
                
                return row['count'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to count keywords for question {question_id}: {e}")
            return 0
