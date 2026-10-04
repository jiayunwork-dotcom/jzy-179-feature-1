"""pytest 共享夹具：独立临时数据目录 + FastAPI TestClient。"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import create_app  # noqa: E402
from app.storage import Storage  # noqa: E402


@pytest.fixture()
def data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture()
def storage(data_dir):
    s = Storage(data_dir)
    yield s
    # 不自动 shutdown 线程池也无妨（临时目录随进程退出）


@pytest.fixture()
def client(storage):
    delay = float(os.environ.get("DIFFUSION_TEST_EVAL_DELAY", "0"))
    app = create_app(storage, eval_delay=delay)
    with TestClient(app) as c:
        yield c
    app.state.jobs.shutdown()
    app.state.burnups.shutdown()


# 常用材料：燃料 k∞=1.2, L²=10
FUEL = dict(thickness=50.0, D=1.0, sigma_a=0.1, nu_sigma_f=0.12, n_mesh=200)
REFLECTOR = dict(thickness=20.0, D=1.0, sigma_a=0.0, nu_sigma_f=0.0, n_mesh=80)


def bare_fuel(a: float = 50.0, n: int = 200, *, D=1.0, sa=0.1, nsf=0.12):
    return dict(thickness=float(a), D=float(D), sigma_a=float(sa),
                nu_sigma_f=float(nsf), n_mesh=int(n))


def make_case(client, regions, *, name="t", left="zero_flux", right="zero_flux",
              settings=None, target_total_fission=1.0):
    body = dict(name=name, regions=regions, left_boundary=left,
                right_boundary=right, target_total_fission=target_total_fission)
    if settings is not None:
        body["settings"] = settings
    r = client.post("/cases", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def solve_case(client, case_id, **kw):
    r = client.post(f"/cases/{case_id}/solve", json=kw)
    assert r.status_code == 200, r.text
    return r.json()


def wait_job(client, job_id, timeout=60.0, step=0.02):
    import time
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/jobs/{job_id}").json()
        if last["status"] not in ("queued", "running"):
            return last
        time.sleep(step)
    raise AssertionError(f"作业 {job_id} 超时未结束: {last}")
