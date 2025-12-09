"""
Aggregator Stage 1: Term Extractor
Bottom-up clustering of novel terms from classified answers.

FLOW:
1. Read all novel terms from high_priority_terms and medium_priority_terms
2. Use LLM to check semantic equivalence between all term pairs
3. Group validated variants into clusters
4. Summarize with frequency, recommendations, and evidence

Uses model configured in .env:
- AGGREGATOR_STAGE1_PROVIDER (openrouter/openai)
- AGGREGATOR_STAGE1_MODEL (e.g., deepseek/deepseek-chat-v3.2)
- AGGREGATOR_STAGE1_TEMPERATURE (0.0 recommended for consistency)

Note: This uses pure LLM reasoning (no embedding pre-filter) for maximum accuracy
in detecting semantic equivalences and novel term patterns.
"""

import sys
from pathlib import Path

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from datetime import datetime
from pydantic_ai import Agent

from agent.models.aggregation import (
    TermVariantValidation,
    TermOccurrence,
    TermCluster,
    Stage1Statistics,
    Stage1Output
)
from agent.config.providers import provider_manager
from agent.prompts.aggregator_stage1 import TERM_VALIDATOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Log which model is configured for this agent
model_info = provider_manager.get_model_info("aggregator_stage1")
logger.info(
    f"Aggregator Stage 1: {model_info['provider'].upper()} - {model_info['model']} "
    f"(temperature={model_info['temperature']})"
)


# ========== PYDANTIC AI AGENT ==========

# Simple validator agent for term equivalence
# No complex tools needed - just validates if two terms mean the same thing
term_validator_agent = Agent(
    model=provider_manager.get_model("aggregator_stage1"),
    result_type=TermVariantValidation,
    system_prompt=TERM_VALIDATOR_SYSTEM_PROMPT
)


# ========== MAIN STAGE 1 LOGIC ==========

class TermExtractor:
    """
    Stage 1: Bottom-up term clustering.
    
    Simpler than Extractor/Classifier:
    - No database tools (works with CSV input)
    - No complex RAG (just embedding clustering + LLM validation)
    - Single responsibility: Group similar terms from student answers
    """
    
    def __init__(self):
        """Initialize term extractor with LLM for semantic reasoning."""
        # No embedding model needed - using pure LLM reasoning for better accuracy
        self.config = provider_manager.get_agent_config("aggregator_stage1")
        logger.info(f"TermExtractor initialized with model: {self.config.model_name}")
        logger.info(f"[Stage 1] Using pure LLM reasoning (no embedding pre-filter)")
        
    async def process(self, classified_df: pd.DataFrame) -> Stage1Output:
        """
        Main entry point for Stage 1 term clustering.
        
        FLOW:
        1. Read: Extract all novel terms from routed answers
        2. Check Semantics: Use LLM to compare all term pairs
        3. Group: Merge LLM-validated equivalent terms into clusters
        4. Summarize: Create clusters with frequency and recommendations
        
        Args:
            classified_df: DataFrame with classifier output, must contain:
                - aggregator_recommendation column (filter for 'ROUTE')
                - high_priority_terms column (pipe-separated)
                - medium_priority_terms column (pipe-separated)
                - answer_id, topic, answer_text columns
        
        Returns:
            Stage1Output with term clusters and statistics
        """
        start_time = datetime.now()
        
        logger.info(f"[Stage 1] === STARTING TERM CLUSTERING ===")
        logger.info(f"[Stage 1] Model: {self.config.model_name} (temp={self.config.temperature})")
        
        # Filter routed answers
        routed = classified_df[
            classified_df['aggregator_recommendation'] == 'ROUTE'
        ].copy()
        
        logger.info(f"[Stage 1] Processing {len(routed)} routed answers...")
        
        if len(routed) == 0:
            logger.warning("[Stage 1] No routed answers found!")
            return self._empty_output()
        
        # STEP 1: READ - Extract all terms from CSV
        logger.info(f"[Stage 1] STEP 1: Reading novel terms from CSV...")
        term_inventory = self._extract_terms(routed)
        logger.info(f"[Stage 1] ✓ Extracted {len(term_inventory)} term occurrences")
        
        if len(term_inventory) == 0:
            logger.warning("[Stage 1] No terms extracted!")
            return self._empty_output()
        
        # STEP 2 & 3: CHECK SEMANTICS + GROUP - Normalize variants using LLM reasoning
        logger.info(f"[Stage 1] STEP 2-3: Checking semantics with LLM and grouping variants...")
        normalized = await self._normalize_terms(term_inventory)
        logger.info(f"[Stage 1] ✓ Normalized to {len(normalized)} canonical terms")
        
        # STEP 4: SUMMARIZE - Create clusters with recommendations
        logger.info(f"[Stage 1] STEP 4: Creating clusters and summarizing...")
        clusters = self._create_clusters(normalized)
        logger.info(f"[Stage 1] ✓ Created {len(clusters)} term clusters (freq >= 2)")
        
        # Enrich with evidence quotes
        enriched = self._enrich_clusters(clusters, routed)
        logger.info(f"[Stage 1] ✓ Enriched clusters with evidence quotes")
        
        # Compute statistics
        stats = self._compute_statistics(enriched, term_inventory)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"[Stage 1] === COMPLETED in {processing_time:.2f}s ===")
        
        return Stage1Output(
            term_clusters=enriched,
            statistics=stats,
            total_terms_processed=len(term_inventory),
            total_answers_processed=len(routed),
            processing_metadata={
                'processing_time_seconds': processing_time,
                'model_used': self.config.model_name,
                'provider': self.config.provider.value,
                'temperature': self.config.temperature,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def _extract_terms(self, routed_df: pd.DataFrame) -> List[Dict]:
        """
        STEP 1 - READ: Extract all novel terms from classifier output.
        
        Reads from:
        - high_priority_terms column (pipe-separated)
        - medium_priority_terms column (pipe-separated)
        
        Returns:
            List of term occurrences with metadata (term, answer_id, priority, topic)
        """
        term_inventory = []
        high_count = 0
        medium_count = 0
        
        for idx, row in routed_df.iterrows():
            answer_id = str(row['answer_id'])
            topic = str(row.get('topic', ''))
            
            # Extract high priority terms
            if pd.notna(row.get('high_priority_terms')):
                high_terms_str = str(row['high_priority_terms'])
                for term in high_terms_str.split('|'):
                    term = term.strip()
                    if term and term.lower() not in ['nan', 'none', '']:
                        term_inventory.append({
                            'term': term,
                            'answer_id': answer_id,
                            'priority': 'high',
                            'topic': topic
                        })
                        high_count += 1
            
            # Extract medium priority terms
            if pd.notna(row.get('medium_priority_terms')):
                medium_terms_str = str(row['medium_priority_terms'])
                for term in medium_terms_str.split('|'):
                    term = term.strip()
                    if term and term.lower() not in ['nan', 'none', '']:
                        term_inventory.append({
                            'term': term,
                            'answer_id': answer_id,
                            'priority': 'medium',
                            'topic': topic
                        })
                        medium_count += 1
        
        logger.info(f"[Stage 1]   High priority: {high_count}, Medium priority: {medium_count}")
        return term_inventory
    
    async def _normalize_terms(self, term_inventory: List[Dict]) -> List[Dict]:
        """
        STEP 2-3: CHECK SEMANTICS + GROUP variants using pure LLM reasoning.
        
        Strategy:
        1. Group identical terms (case-insensitive)
        2. For each term, check ALL other terms with LLM for semantic equivalence
        3. Merge validated variants into canonical clusters
        
        Note: This uses LLM for all comparisons to maximize accuracy in detecting
        semantic equivalences, including subtle variations and novel terminology.
        
        Returns:
            List of normalized term groups with canonical term and variants
        """
        # Group by lowercase for initial deduplication
        term_groups = {}
        for item in term_inventory:
            key = item['term'].lower()
            if key not in term_groups:
                term_groups[key] = []
            term_groups[key].append(item)
        
        logger.info(f"[Stage 1]   Unique terms (case-insensitive): {len(term_groups)}")
        
        # Find variants using LLM reasoning for all pairs
        normalized = []
        processed_terms = set()
        all_terms = list(term_groups.keys())
        llm_validations = 0
        total_pairs = len(all_terms) * (len(all_terms) - 1) // 2
        
        logger.info(f"[Stage 1]   Will check up to {total_pairs} term pairs with LLM...")
        
        for idx, term in enumerate(all_terms):
            if term in processed_terms:
                continue
            
            if (idx + 1) % 5 == 0:
                logger.info(f"[Stage 1]   Processed {idx + 1}/{len(all_terms)} terms ({llm_validations} LLM calls)...")
            
            # Find all variants using LLM validation (no embedding pre-filter)
            variants, validation_count = await self._find_variants_llm_only(term, all_terms, processed_terms)
            llm_validations += validation_count
            
            # Merge all variant occurrences
            all_occurrences = []
            for variant in variants:
                all_occurrences.extend(term_groups[variant])
                processed_terms.add(variant)
            
            # Choose canonical term (most frequent variant)
            variant_counts = {}
            for occ in all_occurrences:
                original = occ['term']
                variant_counts[original] = variant_counts.get(original, 0) + 1
            
            canonical = max(variant_counts.keys(), key=lambda t: variant_counts[t])
            
            normalized.append({
                'canonical_term': canonical,
                'variants': list(set([o['term'] for o in all_occurrences])),
                'occurrences': all_occurrences,
                'frequency': len(all_occurrences),
                'unique_students': len(set([o['answer_id'] for o in all_occurrences]))
            })
        
        logger.info(f"[Stage 1]   LLM validations performed: {llm_validations}")
        return normalized
    
    async def _find_variants_llm_only(self, term: str, all_terms: List[str], 
                                     processed: Set[str]) -> tuple[List[str], int]:
        """
        Find semantically similar terms using pure LLM reasoning (no embedding filter).
        
        This method checks ALL unprocessed terms against the base term using LLM,
        providing maximum accuracy for detecting semantic equivalences.
        
        Args:
            term: Base term to find variants for
            all_terms: All available terms
            processed: Already processed terms (skip these)
        
        Returns:
            Tuple of (variant terms including base term, number of LLM validations performed)
        """
        variants = [term]
        validation_count = 0
        
        # Check against ALL other unprocessed terms using LLM
        for other_term in all_terms:
            if other_term == term or other_term in processed:
                continue
            
            # Use LLM to validate semantic equivalence
            validation_count += 1
            if await self._llm_validates_equivalence(term, other_term):
                variants.append(other_term)
                logger.info(f"[Stage 1] ✓ Grouped: '{term}' ≈ '{other_term}'")
        
        return variants, validation_count
    
    async def _llm_validates_equivalence(self, term1: str, term2: str) -> bool:
        """
        Use LLM to validate if two terms are semantically equivalent.
        
        Uses the model configured in .env (AGGREGATOR_STAGE1_MODEL).
        Temperature from .env ensures consistency across validations.
        
        Returns:
            True if LLM confirms equivalence, False otherwise.
        """
        prompt = f"""Are these terms semantically equivalent in AI/ML context?

Term 1: "{term1}"
Term 2: "{term2}"

Consider: synonyms, abbreviations, hyphenation, technical vs colloquial usage."""

        try:
            # Run with temperature from .env config
            result = await term_validator_agent.run(prompt)
            
            logger.debug(
                f"[Stage 1] LLM ({self.config.model_name}): "
                f"'{term1}' ≈ '{term2}' → {result.data.are_equivalent} "
                f"(reason: {result.data.reasoning})"
            )
            return result.data.are_equivalent
            
        except Exception as e:
            logger.error(f"[Stage 1] LLM validation failed for '{term1}' vs '{term2}': {e}")
            return False
    
    def _create_clusters(self, normalized: List[Dict]) -> List[TermCluster]:
        """
        STEP 4 - SUMMARIZE: Create TermCluster objects with recommendations.
        
        Filters out single occurrences (noise) and creates clusters with:
        - Frequency counts (total occurrences)
        - Unique student counts
        - Recommendations based on adoption:
          * ADD_TO_KB: ≥5 students (strong pattern)
          * REVIEW: 3-4 students (emerging pattern)
          * MONITOR: <3 students (watch for growth)
        
        Returns:
            List of TermCluster objects (only terms with frequency >= 2)
        """
        clusters = []
        skipped_count = 0
        
        for idx, term_data in enumerate(normalized):
            # Skip single occurrences (noise)
            if term_data['frequency'] < 2:
                logger.debug(
                    f"[Stage 1] Skipping low-frequency: {term_data['canonical_term']} (freq=1)"
                )
                skipped_count += 1
                continue
            
            # Count priority breakdown
            priority_breakdown = {'high': 0, 'medium': 0}
            for occ in term_data['occurrences']:
                priority_breakdown[occ['priority']] += 1
            
            # Determine recommendation based on unique students
            unique_students = term_data['unique_students']
            if unique_students >= 5:
                recommendation = "ADD_TO_KB"
                reason = f"{unique_students} students using similar terms - strong pattern"
            elif unique_students >= 3:
                recommendation = "REVIEW"
                reason = f"{unique_students} students - emerging pattern worth reviewing"
            else:
                recommendation = "MONITOR"
                reason = f"Only {unique_students} students - monitor for growth"
            
            # Extract topics
            topics = list(set([
                occ['topic'] for occ in term_data['occurrences'] 
                if occ['topic'] and occ['topic'] != 'nan'
            ]))
            
            cluster = TermCluster(
                cluster_id=f"TC{idx+1:03d}",
                canonical_term=term_data['canonical_term'],
                variants=term_data['variants'],
                frequency=term_data['frequency'],
                unique_students=unique_students,
                student_ids=[o['answer_id'] for o in term_data['occurrences']],
                topics=topics,
                evidence_quotes=[],  # Will be filled in _enrich_clusters
                recommendation=recommendation,
                recommendation_reason=reason,
                priority_breakdown=priority_breakdown
            )
            
            clusters.append(cluster)
        
        if skipped_count > 0:
            logger.info(f"[Stage 1]   Skipped {skipped_count} low-frequency terms (freq < 2)")
        
        return clusters
    
    def _enrich_clusters(self, clusters: List[TermCluster], 
                        routed_df: pd.DataFrame) -> List[TermCluster]:
        """
        Add evidence quotes to clusters by extracting relevant snippets from answers.
        """
        for cluster in clusters:
            # Sample up to 2 different students for evidence
            sample_ids = cluster.student_ids[:2]
            
            for answer_id in sample_ids:
                answer_row = routed_df[routed_df['answer_id'].astype(str) == answer_id]
                
                if not answer_row.empty:
                    answer_text = str(answer_row.iloc[0].get('answer_text', ''))
                    
                    # Try to find any variant of the term in the answer
                    found_term = None
                    for variant in cluster.variants:
                        if variant.lower() in answer_text.lower():
                            found_term = variant
                            break
                    
                    if found_term:
                        quote = self._extract_quote(answer_text, found_term)
                        
                        if quote:
                            cluster.evidence_quotes.append({
                                'answer_id': answer_id,
                                'term': found_term,
                                'quote': quote
                            })
        
        return clusters
    
    def _extract_quote(self, text: str, term: str, context_chars: int = 150) -> str:
        """
        Extract a quote from text containing the term with surrounding context.
        
        Args:
            text: Full answer text
            term: Term to find
            context_chars: Characters of context before/after term
        
        Returns:
            Quote string with ellipsis markers
        """
        term_lower = term.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(term_lower)
        if idx == -1:
            return ""
        
        # Find sentence boundaries (or use character window)
        start = max(0, idx - context_chars)
        end = min(len(text), idx + len(term) + context_chars)
        
        quote = text[start:end].strip()
        
        # Add ellipsis markers
        if start > 0:
            quote = "..." + quote
        if end < len(text):
            quote = quote + "..."
        
        return quote
    
    def _compute_statistics(self, clusters: List[TermCluster], 
                           term_inventory: List[Dict]) -> Stage1Statistics:
        """Compute summary statistics for the analysis."""
        
        # Count priority breakdown
        high_priority = sum(1 for t in term_inventory if t['priority'] == 'high')
        medium_priority = sum(1 for t in term_inventory if t['priority'] == 'medium')
        
        # Count unique students across all clusters
        all_student_ids = set()
        for cluster in clusters:
            all_student_ids.update(cluster.student_ids)
        
        return Stage1Statistics(
            total_clusters=len(clusters),
            add_to_kb_count=sum(1 for c in clusters if c.recommendation == 'ADD_TO_KB'),
            review_count=sum(1 for c in clusters if c.recommendation == 'REVIEW'),
            monitor_count=sum(1 for c in clusters if c.recommendation == 'MONITOR'),
            total_unique_students=len(all_student_ids),
            high_priority_terms=high_priority,
            medium_priority_terms=medium_priority
        )
    
    def _empty_output(self) -> Stage1Output:
        """Return empty output structure when no data to process."""
        return Stage1Output(
            term_clusters=[],
            statistics=Stage1Statistics(
                total_clusters=0,
                add_to_kb_count=0,
                review_count=0,
                monitor_count=0,
                total_unique_students=0,
                high_priority_terms=0,
                medium_priority_terms=0
            ),
            total_terms_processed=0,
            total_answers_processed=0,
            processing_metadata={
                'timestamp': datetime.now().isoformat(),
                'note': 'No data to process'
            }
        )


# ========== CONVENIENCE FUNCTION ==========

async def extract_term_clusters(classified_df: pd.DataFrame) -> Stage1Output:
    """
    Convenience function to run Stage 1 term extraction.
    
    Args:
        classified_df: DataFrame with classifier output
    
    Returns:
        Stage1Output with term clusters
    """
    extractor = TermExtractor()
    return await extractor.process(classified_df)
