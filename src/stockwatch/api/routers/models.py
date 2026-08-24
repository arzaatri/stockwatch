"""GET /models/current - proxies the inference service's own /model/current
so callers never need to know that service exists as a separate thing."""

from fastapi import APIRouter

from stockwatch.api import inference_client
from stockwatch.inference_service.schemas import ModelStatus

router = APIRouter()


@router.get("/models/current", response_model=ModelStatus)
def models_current() -> ModelStatus:
    return inference_client.get_model_status()
