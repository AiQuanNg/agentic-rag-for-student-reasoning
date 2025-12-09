"""
Pydantic models for Aggregator Stage 1 (Term Clustering).

Defines structured outputs for term extraction, clustering, and analysis.
Stage 2 models (theme synthesis) will be added in future implementation.
"""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator


# ========== STAGE 1: TERM CLUSTERING MODELS ==========

class TermVariantValidation(BaseModel):
    """
    LLM output for validating if two terms are semantically equivalent.
    Used by Stage 1 to group term variants.
    """
    are_equivalent: bool = Field(
        ..., 
        description="TRUE if terms are semantically equivalent in AI/ML context, FALSE otherwise"
    )
    reasoning: str = Field(
        ..., 
        description="Brief explanation of why terms are or aren't equivalent"
    )


class TermOccurrence(BaseModel):
    """Single occurrence of a term in a student answer."""
    term: str = Field(..., description="The exact term as written by student")
    answer_id: str = Field(..., description="Student answer ID")
    priority: Literal["high", "medium"] = Field(..., description="Priority level from classifier")
    topic: Optional[str] = Field(default="", description="Topic context where term appears")


class TermCluster(BaseModel):
    """
    A cluster of semantically similar terms after grouping.
    Represents the aggregated view of how students express a concept.
    """
    cluster_id: str = Field(..., description="Unique cluster identifier (e.g., TC001)")
    canonical_term: str = Field(
        ..., 
        description="Most representative/frequent term chosen as cluster name"
    )
    variants: List[str] = Field(
        ..., 
        description="All term variations found in this cluster"
    )
    frequency: int = Field(
        ..., 
        description="Total occurrences across all variants and students"
    )
    unique_students: int = Field(
        ..., 
        description="Number of unique students using any variant of this term"
    )
    student_ids: List[str] = Field(
        ..., 
        description="List of all student answer IDs using this term cluster"
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Topics/contexts where this term cluster appears"
    )
    evidence_quotes: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Sample quotes showing term usage. Each dict has: answer_id, term, quote"
    )
    recommendation: Literal["ADD_TO_KB", "REVIEW", "MONITOR"] = Field(
        ..., 
        description="Action recommendation based on frequency thresholds"
    )
    recommendation_reason: str = Field(
        ..., 
        description="Explanation for the recommendation"
    )
    priority_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of high vs medium priority occurrences"
    )


class Stage1Statistics(BaseModel):
    """Summary statistics for Stage 1 processing."""
    total_clusters: int = Field(..., description="Number of term clusters created")
    add_to_kb_count: int = Field(..., description="Clusters recommended for knowledge base")
    review_count: int = Field(..., description="Clusters needing professor review")
    monitor_count: int = Field(..., description="Low-frequency clusters to monitor")
    total_unique_students: int = Field(..., description="Total unique students in analysis")
    high_priority_terms: int = Field(..., description="Count of high-priority terms processed")
    medium_priority_terms: int = Field(..., description="Count of medium-priority terms processed")


class Stage1Output(BaseModel):
    """
    Complete output from Stage 1 Term Extractor.
    This is what gets passed to Stage 3 (Curator) for final professor review.
    """
    term_clusters: List[TermCluster] = Field(
        ..., 
        description="All term clusters discovered through semantic grouping"
    )
    statistics: Stage1Statistics = Field(
        ..., 
        description="Summary statistics for this analysis"
    )
    total_terms_processed: int = Field(
        ..., 
        description="Total raw term occurrences before clustering"
    )
    total_answers_processed: int = Field(
        ..., 
        description="Total routed answers analyzed"
    )
    processing_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about processing (timestamp, model used, etc.)"
    )
    
    @field_validator('term_clusters')
    @classmethod
    def validate_clusters_not_empty(cls, v: List[TermCluster]) -> List[TermCluster]:
        """Ensure at least some clusters were created (unless no data)."""
        # Allow empty for no-data cases, but could add logic here
        return v


# ========== STAGE 2: THEME SYNTHESIS MODELS (PLACEHOLDER) ==========
# To be implemented in future PR

class Stage2Output(BaseModel):
    """Placeholder for Stage 2 theme synthesis output."""
    themes: List[Dict] = Field(default_factory=list, description="Discovered conceptual themes")
    statistics: Dict = Field(default_factory=dict, description="Theme statistics")
