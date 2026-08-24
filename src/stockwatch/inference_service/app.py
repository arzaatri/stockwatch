"""The isolation-forest microservice's FastAPI app. Run via `stockwatch
serve-inference`. Stateless/DB-agnostic by design - the only thing it reads
from disk is the trained model artifact (detection/model_store.py).
"""

from fastapi import FastAPI

from stockwatch.inference_service.routes import router

app = FastAPI(title="stockwatch-inference", version="0.1.0")
app.include_router(router)
