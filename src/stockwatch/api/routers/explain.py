"""POST /explain - async LLM explanation, since (unlike /detect) it's
seconds-and-dollars, not sub-millisecond. Jobs are Postgres-backed
(api/jobs.py) so polling GET /explain/{job_id} works regardless of which
api replica a given request lands on.
"""

from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException

from stockwatch.api import jobs
from stockwatch.api.schemas import ExplainJobCreated, ExplainJobStatus, ExplainRequest
from stockwatch.logging_utils import get_logger
from stockwatch.pipeline.explain_anomaly import explain_anomaly_at

logger = get_logger(__name__)
router = APIRouter()


@router.post("/explain", response_model=ExplainJobCreated, status_code=202)
def create_explanation(
    request: ExplainRequest, background_tasks: BackgroundTasks
) -> ExplainJobCreated:
    job_id = jobs.create_job(request.ticker, request.window_end)
    background_tasks.add_task(
        _run_explanation,
        job_id,
        request.ticker,
        request.window_end,
        request.news_count,
    )
    return ExplainJobCreated(job_id=job_id, status="pending")


@router.get("/explain/{job_id}", response_model=ExplainJobStatus)
def get_explanation(job_id: str) -> ExplainJobStatus:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")
    return ExplainJobStatus(
        job_id=job.job_id,
        status=job.status,
        ticker=job.ticker,
        window_end=job.window_end,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


def _run_explanation(
    job_id: str, ticker: str, window_end: datetime, news_count: int
) -> None:
    """Runs in a threadpool (Starlette's BackgroundTasks offloads sync
    callables automatically), so the blocking DB/LLM calls inside
    explain_anomaly_at() don't stall the event loop for other requests.
    """
    jobs.mark_running(job_id)
    try:
        result = explain_anomaly_at(ticker, window_end, news_count=news_count)
    except Exception:
        logger.exception(
            "Explanation job %s failed for %s @ %s", job_id, ticker, window_end
        )
        jobs.mark_error(job_id, "Explanation failed - see server logs.")
        return
    jobs.mark_done(
        job_id,
        {
            "context": result["context"].model_dump(mode="json"),
            "explanation": result["explanation"].model_dump(mode="json"),
        },
    )
