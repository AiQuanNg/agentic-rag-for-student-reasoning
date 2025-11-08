from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict

class ExtractionResult(BaseModel):
    """
    Output from Extractor Agent for a student answer.
    """
    matched_keywords: List[str] = Field(
        ..., description="Matched codebook terms found in student's answer (semantic check)."
    )
    detected_themes: List[str] = Field(
        ..., description="Broader subject areas or reasoning types present in answer."
    )
    novel_terms: List[str] = Field(
        default_factory=list,
        description="New concepts NOT already in codebook (potential additions)."
    )
    evidence_spans: List[str] = Field(
        default_factory=list,
        description="Quoted answer snippets as key evidence for understanding."
    )
    extraction_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence score (0.0 = low, 1.0 = high) for this extraction run."
    )
    tools_used: Optional[List[str]] = Field(
        default_factory=list,
        description="Names of retrieval tools utilized (codebook, vector_search, etc.)."
    )
    
    @field_validator('extraction_confidence')
    @classmethod
    def check_confidence_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0 and 1.")
        return v

class ExtractorContext(BaseModel):
    """
    Context/configuration for Extractor Agent calls.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    question_id: int = Field(..., description="Unique question identifier.")
    db_pool: Optional[Any] = Field(None, description="Async database pool (if in use).")
