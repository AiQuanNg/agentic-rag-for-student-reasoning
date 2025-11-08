# Extractor Agent - Ottomator Adaptation Notes

## Ottomator Architecture Insights
- Agent Layer: Pydantic AI with OpenAIModel
- Tools Layer: Functions decorated with @agent.tool
- Dependency Injection: RunContext[ContextType]
- System Prompts: Explicit tool usage guidelines

## Our Adaptations
- Skip knowledge graph (Neo4j) - use PostgreSQL only
- Extractor uses string-based keyword retrieval (simple)
- Classifier will use pgvector retrieval (complex)
- Single API (Orchestrator) instead of separate agent APIs

## Key Patterns to Copy
1. Pydantic AI Agent initialization
2. Tool decorator pattern
3. Context-based dependency injection
4. Structured output validation with Pydantic
