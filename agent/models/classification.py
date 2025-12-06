"""
Pydantic models for Classifier Agent.

Defines classification result models, input models, and validation logic
for classifying student answers as Standard, Latent, or Off-topic.

Follows ottomator pattern for structured validation with Pydantic v2.
Adapted from extraction.py for consistency across agents.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ClassificationResult(BaseModel):
    """
    Output from Classifier Agent for a student answer.
    
    This model represents the final classification decision with supporting
    evidence, reasoning, and context from both Extractor and Classifier.
    
    Follows ottomator ChunkResult pattern for structured search results.
    Adapted from ExtractionResult pattern for consistency.
    """
    
    # ========== PRIMARY CLASSIFICATION ==========
    
    label: Literal["standard", "latent", "off_topic"] = Field(
        ...,
        description="Classification label: standard (uses expected terminology), "
                    "latent (demonstrates understanding through non-standard reasoning), "
                    "or off_topic (lacks relevant understanding)"
    )
    
    classification_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 = uncertain, 1.0 = highly confident) "
                    "for this classification decision"
    )
    
    # ========== EVIDENCE & REASONING ==========
    
    evidence_spans: List[str] = Field(
        default_factory=list,
        description="Exact quotes from student answer supporting the classification decision"
    )
    
    reasoning: str = Field(
        ...,
        description="2-4 sentence explanation of why this classification was chosen, "
                    "referencing both Extractor findings and rubric alignment"
    )
    
    # ========== TWO-LAYER STRUCTURE (New in v4.0) ==========
    
    layer_1_rubric_grading: Dict[str, Any] = Field(
        default_factory=dict,
        description="Layer 1: Pure rubric grading (categorical: 0/50/100). "
                    "Keys: level_achieved (int: 0|50|100), rubric_evidence (list), "
                    "latent_eligible (bool), grading_reasoning (str)"
    )
    
    layer_2_latent_detection: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Layer 2: Latent signal detection (only present if level_achieved=100). "
                    "Keys: signals_breakdown (dict with mechanism_score, novel_terms_score, etc.), "
                    "total_latent_score (float 0-1), classification (latent|standard), "
                    "latent_confidence (high|medium|null), signal_evidence (dict)"
    )
    
    layer_3_novel_terms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Layer 3: Novel term flagging (empty for OFF_TOPIC). "
                    "Each term: term (str), importance_score (float), priority (HIGH|MEDIUM|LOW), "
                    "component_scores (dict), evidence_spans (list), usage_context (str), "
                    "rubric_contribution (str)"
    )
    
    classification_reasoning: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reasoning summary for all three layers. "
                    "Keys: layer_1_summary, layer_2_summary, layer_3_summary, final_decision"
    )
    
    # ========== RUBRIC CONTEXT (Legacy - for backward compatibility) ==========
    
    rubric_assessment: Dict[str, Any] = Field(
        default_factory=dict,
        description="LEGACY: Maps to layer_1_rubric_grading for backward compatibility. "
                    "Example: {'matches_level_100': True, 'rubric_level_achieved': '100', 'rubric_reasoning': '...'}"
    )
    
    # ========== EXTRACTOR INTEGRATION ==========
    
    extractor_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of Extractor findings used in classification. "
                    "Example: {'keywords_used': 4, 'themes_present': ['mechanism'], 'novel_concepts': 1}"
    )
    
    # ========== CRITERIA ASSESSMENT ==========
    
    criteria_assessment: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Which Standard vs Latent indicators were found in the answer. "
                    "Example: {'standard_indicators': ['uses technical terms'], 'latent_indicators': []}"
    )
    
    # ========== LATENT SIGNALS (Legacy - for backward compatibility) ==========
    
    latent_signals: Dict[str, Any] = Field(
        default_factory=dict,
        description="LEGACY: Detailed latent reasoning signals found in the answer. "
                    "Example: {'mechanism_explanations': [...], 'novel_terminology': [...]}"
    )
    
    latent_signals_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="LEGACY: Maps to layer_2_latent_detection.signal_evidence for backward compatibility. "
                    "Keys: mechanism_explanations (list of quotes beyond Level 100), "
                    "novel_terms_in_mechanisms (list of terms used in explanations), "
                    "critical_engagement (quote showing analysis or empty string), "
                    "evidence_quality (specific|general|none), "
                    "cross_domain_connections (quote or empty string)"
    )
    
    # ========== NOVEL TERM FLAGGING (Legacy - for backward compatibility) ==========
    
    flagged_novel_terms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="LEGACY: Maps to layer_3_novel_terms for backward compatibility. "
                    "Each term must have: term (str), importance_score (float 0-1), priority (HIGH/MEDIUM/LOW), "
                    "component_scores (dict with specificity/usage_depth/rubric_alignment/frequency), "
                    "evidence_spans (list), usage_context (str). "
                    "REQUIRED: Must evaluate ALL terms from extraction_results.novel_terms"
    )
    
    aggregator_recommendation: Dict[str, Any] = Field(
        default_factory=lambda: {"route_to_aggregator": False, "reason": ""},
        description="Whether to route this answer to Aggregator for theme discovery. "
                    "Dict with 'route_to_aggregator' (bool) and 'reason' (str). "
                    "route_to_aggregator=True means emerging latent pattern worth investigating"
    )
    
    # ========== METADATA ==========
    
    tools_used: Optional[List[str]] = Field(
        default_factory=list,
        description="Names of tools utilized during classification (e.g., 'get_question_rubrics')"
    )
    
    # ========== CUSTOM VALIDATORS ==========
    
    @field_validator('classification_confidence')
    @classmethod
    def check_confidence_range(cls, v: float) -> float:
        """Ensure classification confidence is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"Classification confidence must be between 0.0 and 1.0, got {v}"
            )
        return v
    
    @field_validator('evidence_spans')
    @classmethod
    def validate_evidence_not_empty(cls, v: List[str]) -> List[str]:
        """
        Validate evidence spans.
        
        Note: Empty evidence_spans allowed for low-confidence classifications.
        Future: Could enforce minimum evidence for high-confidence results.
        """
        # Placeholder for future validation logic
        # Example: if confidence > 0.8 and len(v) == 0: raise error
        return v
    
    def model_post_init(self, __context):
        """
        Post-initialization to ensure backward compatibility.
        
        Populates legacy fields from new layer structure if they're empty.
        This allows the LLM to return only the new layer structure while
        maintaining compatibility with existing CSV export code.
        """
        # Populate rubric_assessment from layer_1_rubric_grading if empty
        if not self.rubric_assessment and self.layer_1_rubric_grading:
            level_achieved = self.layer_1_rubric_grading.get('level_achieved', 0)
            self.rubric_assessment = {
                'level_100_met': level_achieved == 100,
                'level_50_met': level_achieved >= 50,
                'level_0_met': level_achieved >= 0,
                'rubric_level_achieved': str(level_achieved),
                'rubric_evidence': self.layer_1_rubric_grading.get('rubric_evidence', []),
                'latent_eligible': self.layer_1_rubric_grading.get('latent_eligible', False),
                'reasoning': self.layer_1_rubric_grading.get('grading_reasoning', '')
            }
        
        # Populate latent_signals_summary from layer_2_latent_detection if empty
        if not self.latent_signals_summary and self.layer_2_latent_detection:
            signal_evidence = self.layer_2_latent_detection.get('signal_evidence', {})
            self.latent_signals_summary = {
                'mechanism_explanations': signal_evidence.get('mechanism_explanations', []),
                'novel_terms_in_mechanisms': signal_evidence.get('novel_terms_in_mechanisms', []),
                'critical_engagement': signal_evidence.get('critical_engagement', ''),
                'evidence_quality': signal_evidence.get('evidence_quality', 'none'),
                'cross_domain_connections': signal_evidence.get('cross_domain_connections', '')
            }
        
        # Populate flagged_novel_terms from layer_3_novel_terms if empty
        if not self.flagged_novel_terms and self.layer_3_novel_terms:
            self.flagged_novel_terms = self.layer_3_novel_terms


class ClassificationInput(BaseModel):
    """
    Input to Classifier Agent containing student answer and Extractor results.
    
    This model bundles everything the Classifier needs to make a decision:
    - Original student answer text
    - Extractor's findings (keywords, themes, novel terms)
    - Question context
    
    Follows ottomator SearchRequest pattern for input validation.
    """
    
    # ========== CORE INPUT ==========
    
    answer_text: str = Field(
        ...,
        description="Original student answer text to classify"
    )
    
    question_id: int = Field(
        ...,
        description="Question ID for rubric retrieval"
    )
    
    # ========== QUESTION CONTEXT (Fetched upstream, passed directly) ==========
    question_text: Optional[str] = Field(
        None,
        description="Question text for context (e.g., 'What is generative AI?')"
    )
    question_goal: Optional[str] = Field(
        None,
        description="Learning goal (e.g., 'Learn about generative AI basics')"
    )
    question_topic: Optional[str] = Field(
        None,
        description="Topic area (e.g., 'fundamentals', 'application', 'ethics')"
    )

    # ========== EXTRACTOR RESULTS ==========
    
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords found by Extractor Agent"
    )
    
    detected_themes: List[str] = Field(
        default_factory=list,
        description="Themes identified by Extractor Agent"
    )
    
    novel_terms: List[str] = Field(
        default_factory=list,
        description="Novel concepts identified by Extractor Agent"
    )
    
    extraction_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Extractor's confidence in its extraction"
    )
    
    # ========== OPTIONAL CONTEXT ==========
    
    extraction_reasoning: Optional[str] = Field(
        None,
        description="Optional: Extractor's reasoning (if available)"
    )


class ClassifierDependencies(BaseModel):
    """
    Dependencies for Classifier Agent tools.
    
    Provides database access and question context to tools like
    get_question_rubrics() and get_classification_criteria().
    
    Follows ottomator AgentDependencies pattern for dependency injection.
    Matches ExtractorContext pattern from extraction.py.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    question_id: int = Field(
        ...,
        description="Question ID for rubric/criteria retrieval"
    )
    
    db_pool: Optional[Any] = Field(
        None,
        description="Async database connection pool (asyncpg.Pool)"
    )
