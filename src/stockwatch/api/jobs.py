"""Postgres-backed store for async /explain jobs (db/models.py's
ExplanationJob). Deliberately not an in-memory dict: the api service runs
multiple replicas behind a k8s Service, so a poll request can land on a
different pod than the one that created the job - only a shared, durable
store makes that work correctly.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from stockwatch.db.engine import session_scope
from stockwatch.db.models import ExplanationJob


def create_job(ticker: str, window_end: datetime) -> str:
    job_id = str(uuid.uuid4())
    with session_scope() as session:
        session.add(
            ExplanationJob(
                job_id=job_id,
                ticker=ticker,
                window_end=window_end,
                status="pending",
                created_at=datetime.now(UTC),
            )
        )
    return job_id


def get_job(job_id: str) -> ExplanationJob | None:
    with session_scope() as session:
        job = session.get(ExplanationJob, job_id)
        if job is not None:
            session.expunge(job)
        return job


def mark_running(job_id: str) -> None:
    _update(job_id, status="running")


def mark_done(job_id: str, result: dict[str, Any]) -> None:
    _update(job_id, status="done", result=result, completed_at=datetime.now(UTC))


def mark_error(job_id: str, error: str) -> None:
    _update(job_id, status="error", error=error, completed_at=datetime.now(UTC))


def _update(job_id: str, **fields: Any) -> None:
    with session_scope() as session:
        job = session.get(ExplanationJob, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
