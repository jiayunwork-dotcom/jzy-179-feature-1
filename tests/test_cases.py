"""工况一等对象：新建、改某一区、查询、删除、版本留档、重启持久化。"""
from __future__ import annotations

from conftest import Storage, bare_fuel, make_case, solve_case  # noqa: F401
from fastapi.testclient import TestClient


def test_case_crud(client):
    case = make_case(client, [bare_fuel(50.0, 100)], name="首块")
    cid = case["id"]
    assert case["current_version"] == 1
    assert case["regions"][0]["thickness"] == 50.0

    got = client.get(f"/cases/{cid}").json()
    assert got["name"] == "首块"

    listed = client.get("/cases").json()
    assert any(c["id"] == cid for c in listed)

    assert client.delete(f"/cases/{cid}").status_code == 204
    assert client.get(f"/cases/{cid}").status_code == 404
    assert client.delete(f"/cases/{cid}").status_code == 404


def test_patch_region_creates_version_and_keeps_old(client):
    case = make_case(client, [bare_fuel(50.0, 100)], name="改版")
    cid = case["id"]
    r1 = solve_case(client, cid, mode="cold")

    r = client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"thickness": 60.0}})
    assert r.status_code == 200
    assert r.json()["current_version"] == 2
    assert r.json()["regions"][0]["thickness"] == 60.0
    # 其余字段保持
    assert r.json()["regions"][0]["D"] == 1.0

    # 只改名不产生新版本
    r2 = client.patch(f"/cases/{cid}", json={"name": "改版2"})
    assert r2.json()["current_version"] == 2
    assert r2.json()["name"] == "改版2"

    versions = client.get(f"/cases/{cid}/versions").json()
    assert [v["version"] for v in versions] == [1, 2]
    assert versions[0]["regions"][0]["thickness"] == 50.0
    assert versions[1]["regions"][0]["thickness"] == 60.0

    # 旧版结果仍在，且挂在版本 1
    old = client.get(f"/results/{r1['id']}").json()
    assert old["version"] == 1
    v1_results = client.get(f"/cases/{cid}/results", params={"version": 1}).json()
    assert [x["id"] for x in v1_results] == [r1["id"]]

    # 在版本 2 上算，不覆盖版本 1 的结果
    r2sol = solve_case(client, cid, mode="cold")
    assert r2sol["version"] == 2
    assert len(client.get(f"/cases/{cid}/results",
                          params={"version": 1}).json()) == 1
    assert len(client.get(f"/cases/{cid}/results",
                          params={"version": 2}).json()) == 1


def test_patch_unknown_region_index_404(client):
    case = make_case(client, [bare_fuel(50.0, 50)])
    r = client.patch(f"/cases/{case['id']}", json={
        "region_index": 5, "region": {"thickness": 60.0}})
    assert r.status_code == 404


def test_persistence_across_restart(data_dir, client):
    case = make_case(client, [bare_fuel(50.0, 100)], name="持久")
    cid = case["id"]
    res = solve_case(client, cid, mode="cold")
    client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"D": 1.5}})

    # 用同一目录新建 Storage 与应用（模拟服务重启）
    storage2 = Storage(data_dir)
    from app.api import create_app
    app2 = create_app(storage2)
    with TestClient(app2) as c2:
        got = c2.get(f"/cases/{cid}").json()
        assert got["name"] == "持久"
        assert got["current_version"] == 2
        assert got["regions"][0]["D"] == 1.5
        versions = c2.get(f"/cases/{cid}/versions").json()
        assert versions[0]["regions"][0]["D"] == 1.0
        r_old = c2.get(f"/results/{res['id']}").json()
        assert r_old["k_eff"] == res["k_eff"]
    app2.state.jobs.shutdown()


def test_failed_solve_reports_residual_and_iterations(client):
    # max_iter=1 必然不收敛，必须 409 且带最后残差与迭代次数，不许回半成品
    case = make_case(client, [bare_fuel(200.0, 800)], name="不收敛",
                     settings=dict(k_tol=1e-12, flux_tol=1e-12, max_iter=1))
    r = client.post(f"/cases/{case['id']}/solve", json={"mode": "cold"})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["iterations"] == 1
    assert "k_residual" in detail and "flux_residual" in detail
    assert detail["k"] > 0
    # 失败不留结果
    assert client.get(f"/cases/{case['id']}/results").json() == []
