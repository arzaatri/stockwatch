"""The orchestration api service's FastAPI app. Run via `stockwatch serve`.
Owns Postgres, feature building, news/ratings, the LLM, and job tracking;
talks to the isolation-forest microservice (inference_service/) over HTTP
for anything model-related - see api/inference_client.py.
"""

from fastapi import FastAPI

from stockwatch.api.routers import detect, explain, health, models, monitoring

app = FastAPI(title="stockwatch-api", version="0.1.0")
for _router_module in (health, detect, explain, models, monitoring):
    app.include_router(_router_module.router)
