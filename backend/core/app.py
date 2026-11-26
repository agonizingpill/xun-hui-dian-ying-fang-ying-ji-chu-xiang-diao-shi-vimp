from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI
from sqlalchemy import text

from core import auth
from core.db import metadata, transaction
from core.config import SEED_DEMO
from core.middleware import install
from core.migrations import ensure_columns
from core.queueing import broker, view
from domain.quality_routes import quality_router, training_router
from domain.routes import routers
from scripts.seed_demo import seed

@asynccontextmanager
async def lifespan(app):
    from core.db import engine
    metadata.create_all(engine)
    ensure_columns(engine)
    from domain.quality import ensure_registry
    with transaction() as init_conn:
        ensure_registry(init_conn)
    if SEED_DEMO:
        try:
            seed()
        except Exception:
            logging.getLogger('uvicorn.error').exception('demo seed failed')
    yield

app = FastAPI(lifespan=lifespan)
install(app)
app.include_router(auth.router)
app.include_router(training_router)
for prefix in ('/models', '/model', '/quality', '/quality/models', '/quality-model',
               '/quality-models', '/model-versions', '/learning', '/learning/models',
               '/training', '/optics/model', '/optics/models', '/optics/model-versions',
               '/review/models',
               '/reviewer/models', '/api/review/models', '/api/reviewer/models',
               '/api/models', '/api/model', '/api/quality', '/api/quality/models',
               '/api/quality-model', '/api/quality-models', '/api/model-versions'):
    app.include_router(quality_router, prefix=prefix)
for router in routers:
    app.include_router(router)

@app.get('/health')
def health():
    with transaction() as conn:
        conn.execute(text('SELECT 1'))
    return {'ok': bool(broker.ping())}

@app.get('/compute/{token}')
def compute(token: str, user=Depends(auth.actor)):
    return view(user, token)
