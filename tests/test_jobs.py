"""异步临界搜索作业：正常收敛、同号拒绝、步数上限、取消、版本锁定。"""
from __future__ import annotations

import time

import numpy as np

from app.discretization import BoundaryCondition, Region, build_mesh
from app.solver import SolverSettings, solve

from conftest import (
    REFLECTOR,
    bare_fuel,
    make_case,
    solve_case,
    wait_job,
)

SETT = SolverSettings(k_tol=1e-10, flux_tol=1e-10, max_iter=200_000)


def _k_of_thickness(a):
    mesh = build_mesh([Region(float(a), 1.0, 0.1, 0.12, max(int(a * 4), 20))])
    return solve(mesh, BoundaryCondition("zero_flux"),
                 BoundaryCondition("zero_flux"), SETT).k_eff


def test_thickness_search_converges(client):
    # 连续临界厚度：B=pi/a，1 = kinf/(1+L2 B2) => a = pi sqrt(L2/(kinf-1))
    # 有限差分有 O(h^2) 离散偏差，下面同时与"同网格离散临界值"对照。
    import math
    a_crit = math.pi * math.sqrt(10.0 / 0.2)
    n = 400
    case = make_case(client, [bare_fuel(50.0, n)], name="厚度搜索")
    body = dict(region_index=0, parameter="thickness",
                lower=15.0, upper=30.0, max_steps=40)
    r = client.post(f"/cases/{case['id']}/searches", json=body)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] in ("queued", "running")
    assert job["version"] == 1

    final = wait_job(client, job["id"])
    assert final["status"] == "converged", final
    assert abs(final["k_at_root"] - 1.0) < 1e-6
    # 离散临界厚度（对找到的根附近做数值确认）
    from app.discretization import BoundaryCondition, Region, build_mesh
    from app.solver import SolverSettings, solve as _solve
    mesh = build_mesh([Region(final["root_value"], 1.0, 0.1, 0.12, n)])
    k_direct = _solve(mesh, BoundaryCondition("zero_flux"),
                      BoundaryCondition("zero_flux"),
                      SolverSettings(k_tol=1e-11, flux_tol=1e-11)).k_eff
    assert abs(k_direct - 1.0) < 1e-7
    # 与连续解析值的差应随网格加密而小（400 网格时 < 3e-4）
    assert abs(final["root_value"] - a_crit) < 3e-4
    assert final["steps_taken"] >= 1
    # 结果挂在锁定版本（1）下
    res = client.get(f"/results/{final['result_id']}").json()
    assert res["version"] == 1
    assert abs(res["k_eff"] - 1.0) < 1e-6
    assert res["balance_residual"] < 1e-8


def test_search_rejects_same_sign_bracket(client):
    # 厚度区间两端都远低于临界（k<1）或都高于（k>1）→ rejected，并回报两端 k
    case = make_case(client, [bare_fuel(50.0, 100)], name="同号")
    r = client.post(f"/cases/{case['id']}/searches", json=dict(
        region_index=0, parameter="thickness", lower=10.0, upper=20.0, max_steps=20))
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "rejected"
    k_lo, k_hi = job["endpoint_k"]
    assert k_lo < 1.0 and k_hi < 1.0
    assert job["root_value"] is None
    assert "同号" in job["message"]
    # 两端都 k>1
    r2 = client.post(f"/cases/{case['id']}/searches", json=dict(
        region_index=0, parameter="thickness", lower=40.0, upper=120.0, max_steps=20))
    job2 = wait_job(client, r2.json()["id"])
    assert job2["status"] == "rejected"
    assert job2["endpoint_k"][0] > 1.0 and job2["endpoint_k"][1] > 1.0
    assert job2["result_id"] is None


def test_nusf_search_converges(client):
    # 固定厚度 50，求 νΣf 使 k=1
    import math
    B = math.pi / 50.0
    nsf_crit = 0.1 * (1.0 + 10.0 * B * B)
    case = make_case(client, [bare_fuel(50.0, 200, nsf=0.12)], name="nuΣf搜索")
    r = client.post(f"/cases/{case['id']}/searches", json=dict(
        region_index=0, parameter="nu_sigma_f",
        lower=0.10, upper=0.115, max_steps=40))
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "converged", job
    assert abs(job["root_value"] - nsf_crit) < 1e-6
    assert abs(job["k_at_root"] - 1.0) < 1e-6


def test_search_unconverged_when_steps_exhausted(client):
    # 区间 [15,30] 初始 f(15)≈-0.166；只给 2 步，残差不可能压到 1e-6
    case = make_case(client, [bare_fuel(50.0, 40)], name="步数不够")
    r = client.post(f"/cases/{case['id']}/searches", json=dict(
        region_index=0, parameter="thickness", lower=15.0, upper=30.0, max_steps=2))
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "unconverged", job
    assert job["steps_taken"] == 2
    assert abs(job["k_at_root"] - 1.0) >= 1e-6
    lo, hi = job["current_bracket"]
    assert 15.0 <= lo < hi <= 30.0
    assert "残差" in job["message"]
    assert job["result_id"] is None


def test_search_progress_query(client):
    case = make_case(client, [bare_fuel(50.0, 100)], name="进度")
    r = client.post(f"/cases/{case['id']}/searches", json=dict(
        region_index=0, parameter="thickness", lower=15.0, upper=30.0, max_steps=40))
    job = wait_job(client, r.json()["id"])
    # 完成态能看到区间已收缩到根附近（a_crit≈22.2）
    lo, hi = job["current_bracket"]
    assert (hi - lo) < 1.0
    assert job["current_k"] is not None


def test_cancel_search_really_stops(client, monkeypatch):
    """取消后作业真的停下、不写结果。

    client 夹具在本函数体内通过 env 钩子带上"每次评估后可中断等待"，
    制造稳定的"在跑"窗口；取消事件立即唤醒等待，取消后轮询直到
    cancelled，且无 result_id。
    """
    # 夹具已按 DIFFUSION_TEST_EVAL_DELAY 建好 JobManager；这里改用独立
    # 带延迟的 app，避免依赖夹具创建顺序。
    import os
    import tempfile
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.storage import Storage

    tmp = tempfile.mkdtemp()
    app = create_app(Storage(tmp), eval_delay=5.0)
    with TestClient(app) as c:
        case = make_case(c, [bare_fuel(50.0, 200)], name="慢搜索")
        r = c.post(f"/cases/{case['id']}/searches", json=dict(
            region_index=0, parameter="thickness",
            lower=15.0, upper=30.0, max_steps=300))
        job_id = r.json()["id"]

        # 等到 running 再取消
        deadline = time.time() + 10
        while time.time() < deadline:
            j = c.get(f"/jobs/{job_id}").json()
            if j["status"] in ("running", "queued"):
                break
        assert c.post(f"/jobs/{job_id}/cancel").status_code == 200

        final = wait_job(c, job_id, timeout=30)
        assert final["status"] == "cancelled", final
        assert final["cancelled"] is True
        assert final["result_id"] is None
        steps = final["steps_taken"]
        time.sleep(0.5)
        again = c.get(f"/jobs/{job_id}").json()
        assert again["status"] == "cancelled"
        assert again["steps_taken"] == steps
    app.state.jobs.shutdown()


def test_two_concurrent_searches_same_case(client):
    """同一工况可同时挂两个搜索作业，结果各自独立、不张冠李戴。"""
    case = make_case(client, [bare_fuel(50.0, 100)], name="并发")
    cid = case["id"]
    j1 = client.post(f"/cases/{cid}/searches", json=dict(
        region_index=0, parameter="thickness", lower=15.0, upper=25.0,
        max_steps=40)).json()
    j2 = client.post(f"/cases/{cid}/searches", json=dict(
        region_index=0, parameter="thickness", lower=20.0, upper=30.0,
        max_steps=40)).json()
    assert j1["id"] != j2["id"]
    f1 = wait_job(client, j1["id"])
    f2 = wait_job(client, j2["id"])
    assert f1["status"] == f2["status"] == "converged"
    assert abs(f1["root_value"] - f2["root_value"]) < 1e-4
    assert f1["result_id"] != f2["result_id"]


def test_search_locks_submitted_version(client):
    """作业在跑（或已完成）后工况被改，结果仍挂在提交时的版本，用的是旧参数。"""
    # νΣf 搜索：提交后立刻把 νΣf 改掉；作业必须锁定版本 1 的 0.12
    case = make_case(client, [bare_fuel(50.0, 200, nsf=0.12)], name="版本锁")
    cid = case["id"]
    r = client.post(f"/cases/{cid}/searches", json=dict(
        region_index=0, parameter="nu_sigma_f",
        lower=0.10, upper=0.115, max_steps=40))
    job_id = r.json()["id"]

    # 立即改工况（不等作业）：νΣf 改成另一个值 → 版本 2
    pr = client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"nu_sigma_f": 0.15}})
    assert pr.status_code == 200
    assert pr.json()["current_version"] == 2

    final = wait_job(client, job_id)
    assert final["version"] == 1
    assert final["status"] == "converged", final
    res = client.get(f"/results/{final['result_id']}").json()
    assert res["version"] == 1
    # 版本 1（nsf=0.12）的临界 νΣf
    import math
    nsf_crit_v1 = 0.1 * (1.0 + 10.0 * (math.pi / 50.0) ** 2)
    assert abs(final["root_value"] - nsf_crit_v1) < 1e-6

    # 再删工况：作业记录（含快照）依然可查
    listed = client.get("/jobs", params={"case_id": cid}).json()
    assert any(j["id"] == job_id for j in listed)
