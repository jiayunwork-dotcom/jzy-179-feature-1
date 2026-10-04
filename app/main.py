"""应用入口：uvicorn app.main:app"""
from __future__ import annotations

import os

from .api import create_app
from .storage import Storage

DATA_DIR = os.environ.get("DIFFUSION_DATA_DIR", "/data")

storage = Storage(DATA_DIR)
app = create_app(storage)


@app.on_event("shutdown")
def _shutdown() -> None:  # pragma: no cover
    app.state.jobs.shutdown()
    app.state.burnups.shutdown()
