"""
# Extract and structure the content from each student answer:

1. Topic Selection: Identify all distinct topics present in the student’s essay.

2. Topic Relevance Analysis: For each extracted essay topic:

**Semantic Keyword Matching**: Match student concepts to approved codebook keywords
   - Use meaning, not exact word match
   - Example: "transformers" matches "neural networks"
   - Example: "LLMs can write code" matches "code generation"

**Theme Detection**: Identify 1-3 broader themes
   - Options: technical, business, ethical, strategic, implementation

**Novel Terms**: Flag new concepts not in approved codebook
   - Must be substantive (not "important", "useful", "system")

**Evidence Extraction**: Quote 2-4 key phrases showing student's reasoning
   - Must be exact quotes from the answer

3. Confidence Scoring
For each essay topic, compute a confidence score based on:

0.90–1.00: Fully and correctly answers all parts of the question with clear logic and precise semantics.
0.75–0.89: Answers part of the question with clear logic and precise semantics.
0.60–0.74: Provides a simple or partial answer, but contains logical or semantic gaps.
0.40–0.59: Response is related to the question but does not fully address what was asked.
Below 0.40: Little relevant content provided.
"""

## Output Format (JSON only)

{
  "topics":
  	{
  		"Definition of generative AI": 
  			{
  				"extraction_confidence": 0.85,
          "reason:"[]
          "# of matched_keywords": 2,
  				"matched_keywords": ["keyword1", "keyword2"],
  				"# of novel_terms": 1,
  				"novel_terms": ["new_concept"],
  			},
  		"How it gathers information": 
  			{
          "extraction_confidence": 0.85,
          "reason:"[]
  				"# of matched_keywords": 2,
          "matched_keywords": ["keyword1", "keyword2"],
          "# of novel_terms": 1,
          "novel_terms": ["new_concept"],
  			}		
  	},
}

# columns of result
'''
answerID, questionID, topicID, extraction_confidence, reason, #_matched_keywords, matched_keywrods, #_novel_terms, novel_terms
'''
