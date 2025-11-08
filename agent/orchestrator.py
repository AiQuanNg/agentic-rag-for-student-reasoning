"""Orchestrator Agent (Phase 2 - Minimal, orchestration only)

Run this file directly:
    python agent/orchestrator.py --batch-size 10 --question 1

Responsibilities (this file only):
1) Create a run in `runs`.
2) Enqueue submissions from `student_submissions` into `run_items`.
3) Pull small batches and call external agents (imported modules).
4) Persist each item's JSON result into `run_items.result` and mark status.

Agent logic lives in separate modules:
    - agent/extractor.py  → extracting_agent, ExtractorOutput
    - agent/classifier.py → classifier_agent, ClassifierOutput
    - agent/reporter.py   → reporter_agent,  ReporterOutput

Notes:
- Uses asyncpg for DB access and Pydantic for config only.
- Requires tables `runs` and `run_items` (see add_runs.sql).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import asyncio
import json
import os
from typing import Optional, List

import asyncpg
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import separated agents
from agent.extractor import extracting_agent, ExtractorOutput
from agent.classifier import classifier_agent, ClassifierOutput
from agent.reporter import reporter_agent, ReporterOutput


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator run."""

    batch_size: int = Field(10, ge=1)
    question_id: Optional[int] = None


# ----------------------------
# SQL
# ----------------------------

SQL_INSERT_RUN = """
INSERT INTO runs (question_id, total_items, status, config)
VALUES ($1, 0, 'created', $2::jsonb)
RETURNING id;
"""

SQL_UPDATE_RUN_STATUS = """
UPDATE runs SET status = $2, total_items = COALESCE($3, total_items)
WHERE id = $1;
"""

SQL_SELECT_SUBMISSIONS = """
SELECT id FROM student_submissions
WHERE ($1::bigint IS NULL OR question_id = $1)
ORDER BY id
LIMIT $2;
"""

SQL_ENQUEUE_ITEM = """
INSERT INTO run_items (run_id, submission_id, status, attempt_count)
VALUES ($1, $2, 'queued', 0)
ON CONFLICT (run_id, submission_id) DO NOTHING
RETURNING id;
"""

SQL_FETCH_NEXT_BATCH = """
SELECT ri.id, ri.submission_id
FROM run_items ri
JOIN student_submissions s ON s.id = ri.submission_id
WHERE ri.run_id = $1 AND ri.status IN ('queued','processing')
ORDER BY ri.id
LIMIT $2;
"""

SQL_MARK_PROCESSING = """
UPDATE run_items SET status = 'processing', attempt_count = attempt_count + 1
WHERE id = ANY($1::bigint[]);
"""

SQL_GET_SUBMISSION = """
SELECT s.id, s.student_id, s.question_id, s.answer_text
FROM student_submissions s
WHERE s.id = $1;
"""

SQL_MARK_DONE = """
UPDATE run_items
SET status = 'done', result = $2::jsonb
WHERE id = $1;
"""

SQL_MARK_ERROR = """
UPDATE run_items
SET status = 'error', error = $2
WHERE id = $1;
"""


class Orchestrator:
    """Minimal orchestrator that chains external agents.

    This class does not contain agent logic—only orchestration.
    """

    def __init__(self, pool: asyncpg.Pool, cfg: OrchestratorConfig) -> None:
        self.pool = pool
        self.cfg = cfg

    async def create_run(self) -> int:
        async with self.pool.acquire() as conn:
            run_id = await conn.fetchval(
                SQL_INSERT_RUN, self.cfg.question_id, self.cfg.model_dump_json()
            )
            return run_id

    async def enqueue(self, run_id: int) -> int:
        """Enqueue a small window of submissions into run_items.

        Returns the number of enqueued items.
        """
        async with self.pool.acquire() as conn:
            candidates = await conn.fetch(
                SQL_SELECT_SUBMISSIONS, self.cfg.question_id, self.cfg.batch_size
            )
            count = 0
            for row in candidates:
                enq_id = await conn.fetchval(SQL_ENQUEUE_ITEM, run_id, row["id"])
                if enq_id is not None:
                    count += 1
            return count

    async def process(self, run_id: int) -> None:
        """Process items in batches until the queue empties."""
        while True:
            async with self.pool.acquire() as conn:
                batch = await conn.fetch(
                    SQL_FETCH_NEXT_BATCH, run_id, self.cfg.batch_size
                )
                if not batch:
                    break
                item_ids = [b["id"] for b in batch]
                await conn.execute(SQL_MARK_PROCESSING, item_ids)

            for b in batch:
                await self._process_one(b["id"], b["submission_id"])  # noqa: SLF001

    async def _process_one(self, item_id: int, submission_id: int) -> None:
        try:
            async with self.pool.acquire() as conn:
                sub = await conn.fetchrow(SQL_GET_SUBMISSION, submission_id)
            if sub is None:
                raise RuntimeError(f"Submission {submission_id} not found")

            answer_text: str = sub["answer_text"]
            question_id: Optional[int] = sub["question_id"]

            # Agent chain (from separate modules)
            extractor: ExtractorOutput = await extracting_agent(answer_text, question_id)
            classifier: ClassifierOutput = await classifier_agent(answer_text, extractor)
            reporter: ReporterOutput = await reporter_agent(answer_text, classifier)

            result_payload = {
                "submission_id": submission_id,
                "extractor": extractor.model_dump(),
                "classifier": classifier.model_dump(),
                "reporter": reporter.model_dump(),
            }

            async with self.pool.acquire() as conn:
                await conn.execute(SQL_MARK_DONE, item_id, json.dumps(result_payload))
        except Exception as e:  # noqa: BLE001
            async with self.pool.acquire() as conn:
                await conn.execute(SQL_MARK_ERROR, item_id, str(e))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Orchestrator")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--question", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set in environment or .env")

    cfg = OrchestratorConfig(batch_size=args.batch_size, question_id=args.question)
    pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=5)
    try:
        orch = Orchestrator(pool, cfg)
        run_id = await orch.create_run()
        enq = await orch.enqueue(run_id)
        async with pool.acquire() as conn:
            await conn.execute(SQL_UPDATE_RUN_STATUS, run_id, "running", enq)
        await orch.process(run_id)
        async with pool.acquire() as conn:
            await conn.execute(SQL_UPDATE_RUN_STATUS, run_id, "done", None)
        print(f"Run {run_id} complete. Items processed: {enq}")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
