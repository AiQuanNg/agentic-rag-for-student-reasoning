# agent/reporter.py
from pydantic import BaseModel
from typing import Dict, Any

class ReporterOutput(BaseModel):
    summary: str
    notes: Dict[str, Any] = {}

async def reporter_agent(answer_text: str, classifier_output) -> ReporterOutput:
    # Minimal stub for orchestrator testing
    return ReporterOutput(
        summary="Report generated (stub).",
        notes={"classifier": classifier_output.model_dump()},
    )
