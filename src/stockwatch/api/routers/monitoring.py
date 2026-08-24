"""GET /monitoring/drift - stubbed here, wired up to monitoring/drift.py
once that package exists (drift-monitoring checkpoint)."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/monitoring/drift")
def drift() -> None:
    raise HTTPException(status_code=501, detail="Drift monitoring not implemented yet")
