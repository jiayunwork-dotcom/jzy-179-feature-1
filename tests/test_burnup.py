"""燃耗历程：均匀手推对照、零截面不变、单调下降与不变量、对称性、
分段续算（含服务重启）与一次推到底逐点一致、取消保留已完成点、
双版本并发隔离、输入逐项拒收、老数据兼容。
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from fastapi.testclient import TestClient

from app.api import create_app
from app.depletion import (
    BARN_TO_CM2,
    SECONDS_PER_DAY,
    deplete_step,
    depleted_mesh,
    initial_remaining,
    region_remaining,
)
from app.discretization import build_mesh, region_from_dict
from app.physics import uniform_burnup_reference
from app.storage import Storage

from conftest import bare_fuel, make_case, solve_case

# 燃耗声明：σ=500 barn，Σa 的 80% 来自易裂变核素
BURN = dict(initial_density=0.05, micro_sigma_a=500.0,
            fissile_absorption_fraction=0.8)
# 目标总裂变率（"功率旋钮"）：让 100 天烧掉约 36% 易裂变核素
POWER = 5.0e14


def burn_fuel(a=50.0, n=100, *, burnup=BURN, **kw):
    d = bare_fuel(a, n, **kw)
    if burnup is not None:
        d["burnup"] = dict(burnup)
    return d


def make_burn_case(client, regions, *, left="zero_flux", right="zero_flux",
                   target=POWER, name="燃耗"):
    return make_case(client, regions, name=name, left=left, right=right,
                     target_total_fission=target)


def start_burnup(client, case_id, steps, **kw):
    body = {"steps": list(steps)}
    body.update(kw)
    r = client.post(f"/cases/{case_id}/burnups", json=body)
    assert r.status_code == 202, r.text
    return r.json()


def wait_burnup(client, bid, timeout=60.0, step=0.02):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/burnups/{bid}").json()
        if last["status"] != "running":
            return last
        time.sleep(step)
    raise AssertionError(f"燃耗历程 {bid} 超时未结束: {last}")


def get_point(client, bid, n):
    r = client.get(f"/burnups/{bid}/points/{n}")
    assert r.status_code == 200, r.text
    return r.json()


def hand(t_days):
    """均匀全反射算例的手推参考（与本文件算例参数一致）。"""
    return uniform_burnup_reference(
        nu_sigma_f=0.12, sigma_a=0.1, fissile_absorption_fraction=0.8,
        micro_sigma_a=500.0, target_total_fission=POWER, thickness=50.0,
        time_days=t_days)


# ---------- 耗散/截面缩放的库级单元测试 ----------

def test_depletion_library_units():
    regions = [
        region_from_dict(dict(bare_fuel(10.0, 4), burnup=dict(BURN))),
        region_from_dict(dict(thickness=10.0, D=1.0, sigma_a=0.0,
                              nu_sigma_f=0.0, n_mesh=4)),
    ]
    mesh = build_mesh(regions)
    rem = initial_remaining(mesh)
    assert np.all(rem == 1.0)

    phi = np.full(mesh.n, 2.0)
    new = deplete_step(mesh, regions, rem, phi, 10.0)
    expo = 500.0 * BARN_TO_CM2 * 2.0 * 10.0 * SECONDS_PER_DAY
    np.testing.assert_allclose(new[:4], math.exp(-expo), rtol=1e-14)
    assert np.all(new[4:] == 1.0)  # 未声明燃耗的区不动

    m2 = depleted_mesh(mesh, regions, new)
    s = float(new[0])
    np.testing.assert_allclose(m2.nu_sigma_f[:4], 0.12 * s, rtol=1e-14)
    np.testing.assert_allclose(m2.sigma_a[:4], 0.1 * (0.2 + 0.8 * s),
                               rtol=1e-14)
    # D 与未声明区截面不变
    assert np.all(m2.D == mesh.D)
    assert np.all(m2.sigma_a[4:] == mesh.sigma_a[4:])
    assert np.all(m2.nu_sigma_f[4:] == mesh.nu_sigma_f[4:])

    rr = region_remaining(mesh, regions, new)
    assert abs(rr[0] - math.exp(-expo)) < 1e-15
    assert rr[1] == 1.0


# ---------- 1. 均匀全反射：贴住手推值，偏差随步长减半而缩小 ----------

def test_uniform_reflective_matches_hand_derived_and_halving(client):
    case = make_burn_case(client, [burn_fuel(50.0, 50)], name="均匀全反射",
                          left="reflective", right="reflective")
    cid = case["id"]
    ha = start_burnup(client, cid, [10.0] * 10)   # 10 步 × 10 天
    hb = start_burnup(client, cid, [5.0] * 20)    # 20 步 × 5 天（步长减半）
    fa = wait_burnup(client, ha["id"])
    fb = wait_burnup(client, hb["id"])
    assert fa["status"] == fb["status"] == "completed"

    errs_a, errs_b = [], []
    for n in range(0, 11):
        p = get_point(client, ha["id"], n)
        t = 10.0 * n
        s_ref, k_ref = hand(t)
        assert p["time_days"] == t
        assert p["balance_residual"] < 1e-8
        # 每个时间点都要贴住手推值
        assert abs(p["k_eff"] - k_ref) < 1e-2
        assert abs(p["region_remaining"][0] - s_ref) < 1.5e-2
        errs_a.append(abs(p["k_eff"] - k_ref))
    for n in range(0, 21):
        p = get_point(client, hb["id"], n)
        t = 5.0 * n
        s_ref, k_ref = hand(t)
        assert p["balance_residual"] < 1e-8
        assert abs(p["k_eff"] - k_ref) < 6e-3
        assert abs(p["region_remaining"][0] - s_ref) < 8e-3
        errs_b.append(abs(p["k_eff"] - k_ref))

    # 一阶格式：时间步减半，首项误差减半（比值 ≈ 2）
    ratio = max(errs_a) / max(errs_b)
    assert 1.7 < ratio < 2.3, f"时间步减半偏差未减半：比值 {ratio}"
    print(f"\n[均匀全反射燃耗] 10天步长最大k偏差 {max(errs_a):.3e}，"
          f"5天步长 {max(errs_b):.3e}，比值 {ratio:.2f}")


# ---------- 2. 微观截面为零：一切与燃耗前一致 ----------

def test_zero_micro_cross_section_freezes_everything(client):
    burn = dict(BURN, micro_sigma_a=0.0)
    case = make_case(client, [burn_fuel(50.0, 80, burnup=burn)],
                     name="零截面", target_total_fission=POWER,
                     settings=dict(k_tol=1e-11, flux_tol=1e-11))
    h = start_burnup(client, case["id"], [10.0] * 5)
    f = wait_burnup(client, h["id"])
    assert f["status"] == "completed"
    p0 = get_point(client, h["id"], 0)
    for n in range(1, 6):
        p = get_point(client, h["id"], n)
        assert p["region_remaining"][0] == 1.0
        assert all(x == 1.0 for x in p["cell_remaining"])
        # 每个点的 k 与通量都与燃耗前一致（到求解收敛容差）
        assert abs(p["k_eff"] - p0["k_eff"]) < 1e-9
        np.testing.assert_allclose(p["phi"], p0["phi"], rtol=1e-8, atol=1e-10)


# ---------- 3. 只有消耗没有增殖：k 单调下降，份额 ∈ (0,1]，通量非负 ----------

def test_k_decreases_monotonically_and_invariants(client):
    case = make_burn_case(client, [burn_fuel(50.0, 100)])
    h = start_burnup(client, case["id"], [10.0] * 10)
    f = wait_burnup(client, h["id"])
    assert f["status"] == "completed"

    ks, ss = [], []
    for n in range(0, 11):
        p = get_point(client, h["id"], n)
        ks.append(p["k_eff"])
        ss.append(p["region_remaining"][0])
        assert 0.0 < p["region_remaining"][0] <= 1.0
        assert all(0.0 < x <= 1.0 for x in p["cell_remaining"])
        assert all(x >= 0.0 for x in p["phi"])
        assert p["balance_residual"] < 1e-8
        # 每个点都带完整的反应率与两端泄漏
        assert len(p["region_absorption"]) == len(p["region_fission"]) == 1
        assert p["leakage_left"] > 0.0 and p["leakage_right"] > 0.0
        assert abs(p["total_fission"] - POWER) / POWER < 1e-9
    assert all(k2 < k1 for k1, k2 in zip(ks, ks[1:])), "k 未随燃耗单调下降"
    assert all(s2 < s1 for s1, s2 in zip(ss, ss[1:])), "剩余份额未单调下降"

    # 进度字段：已完成步数、当前燃耗时间、当前 k
    assert f["steps_completed"] == 10
    assert f["current_time_days"] == 100.0
    assert f["current_k"] == ks[-1]


# ---------- 4. 左右对称问题燃耗后依旧对称 ----------

def test_symmetric_burnup_stays_symmetric(client):
    case = make_burn_case(client, [burn_fuel(50.0, 100)])
    h = start_burnup(client, case["id"], [20.0] * 5)
    f = wait_burnup(client, h["id"])
    assert f["status"] == "completed"
    p = get_point(client, h["id"], 5)
    phi = np.array(p["phi"])
    rem = np.array(p["cell_remaining"])
    assert np.max(np.abs(phi - phi[::-1])) / phi.max() < 1e-8
    assert np.max(np.abs(rem - rem[::-1])) < 1e-10
    # 中心烧得快、边缘烧得慢
    assert rem[len(rem) // 2] < rem[0]
    assert rem[len(rem) // 2] < rem[-1]


# ---------- 5. 分段推进（中间重启服务）与一次推到底逐点一致 ----------

def test_segmented_matches_one_shot_across_restart(client, data_dir):
    case = make_burn_case(client, [burn_fuel(50.0, 60)])
    cid = case["id"]

    # 第一段：推 4 步后停下（paused）
    h1 = start_burnup(client, cid, [10.0] * 10, max_steps=4)
    f1 = wait_burnup(client, h1["id"])
    assert f1["status"] == "paused"
    assert f1["steps_completed"] == 4
    assert f1["current_time_days"] == 40.0
    assert f1["current_k"] is not None
    assert len(client.get(f"/burnups/{h1['id']}/points").json()) == 5  # 点 0..4

    # 模拟服务重启：同一数据目录新建 Storage 与应用
    storage2 = Storage(data_dir)
    app2 = create_app(storage2)
    with TestClient(app2) as c2:
        got = c2.get(f"/burnups/{h1['id']}").json()
        assert got["status"] == "paused"  # 已停下的历程不受重启影响
        r = c2.post(f"/burnups/{h1['id']}/advance", json={})
        assert r.status_code == 202, r.text
        f1b = wait_burnup(c2, h1["id"])
        assert f1b["status"] == "completed"
        assert f1b["steps_completed"] == 10
        assert len(c2.get(f"/burnups/{h1['id']}/points").json()) == 11

        # 对照：同一版本上一次推 10 步到底
        h2 = start_burnup(c2, cid, [10.0] * 10)
        f2 = wait_burnup(c2, h2["id"])
        assert f2["status"] == "completed"

        for n in range(0, 11):
            p1 = get_point(c2, h1["id"], n)
            p2 = get_point(c2, h2["id"], n)
            assert p1["time_days"] == p2["time_days"]
            # k 之差远在收敛容差量级之下（确定性续算，实际逐位一致）
            assert abs(p1["k_eff"] - p2["k_eff"]) <= 1e-12
            s1, s2 = p1["region_remaining"][0], p2["region_remaining"][0]
            assert abs(s1 - s2) / s2 <= 1e-9
    app2.state.jobs.shutdown()
    app2.state.burnups.shutdown()


# ---------- 6. 取消：已完成的点保留，未完成的步不写入；之后可续算 ----------

def test_cancel_keeps_completed_points_and_resume_finishes(data_dir):
    storage = Storage(data_dir)
    app = create_app(storage, burnup_step_delay=2.0)
    with TestClient(app) as c:
        case = make_burn_case(c, [burn_fuel(50.0, 60)])
        h = start_burnup(c, case["id"], [10.0] * 10)
        bid = h["id"]

        deadline = time.time() + 15
        doc = None
        while time.time() < deadline:
            doc = c.get(f"/burnups/{bid}").json()
            if doc["steps_completed"] >= 1:
                break
            time.sleep(0.02)
        assert doc is not None and doc["steps_completed"] >= 1

        assert c.post(f"/burnups/{bid}/cancel").status_code == 200
        fin = wait_burnup(c, bid, timeout=30)
        assert fin["status"] == "cancelled"
        assert fin["cancelled"] is True
        done = fin["steps_completed"]
        assert 1 <= done < 10
        # 已完成的点保留（步数 + 初始点），未完成的步不写入
        assert len(c.get(f"/burnups/{bid}/points").json()) == done + 1
        time.sleep(0.5)  # 线程真的停了：点数不再增长
        assert len(c.get(f"/burnups/{bid}/points").json()) == done + 1
    app.state.jobs.shutdown()
    app.state.burnups.shutdown()

    # “重启”之后接着往后推，直到完成
    storage2 = Storage(data_dir)
    app2 = create_app(storage2)
    with TestClient(app2) as c2:
        r = c2.post(f"/burnups/{bid}/advance", json={})
        assert r.status_code == 202, r.text
        fin2 = wait_burnup(c2, bid)
        assert fin2["status"] == "completed"
        assert fin2["steps_completed"] == 10
        assert len(c2.get(f"/burnups/{bid}/points").json()) == 11
    app2.state.jobs.shutdown()
    app2.state.burnups.shutdown()


# ---------- 7. 两条历程挂在同一工况的不同版本上并发推进，互不混入 ----------

def test_two_histories_on_two_versions_concurrently(client):
    case = make_burn_case(client, [burn_fuel(50.0, 60)])
    cid = case["id"]
    h1 = start_burnup(client, cid, [10.0] * 6)      # 绑定版本 1

    # 立刻改燃耗参数 → 版本 2（h1 在跑/已完成都不受影响）
    burn2 = dict(BURN, fissile_absorption_fraction=0.5)
    r = client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"burnup": burn2}})
    assert r.status_code == 200, r.text
    assert r.json()["current_version"] == 2

    h2 = start_burnup(client, cid, [10.0] * 6)      # 绑定版本 2（当前版）
    f1 = wait_burnup(client, h1["id"])
    f2 = wait_burnup(client, h2["id"])
    assert f1["version"] == 1 and f2["version"] == 2
    assert f1["status"] == f2["status"] == "completed"

    pts1 = client.get(f"/burnups/{h1['id']}/points").json()
    pts2 = client.get(f"/burnups/{h2['id']}/points").json()
    assert len(pts1) == len(pts2) == 7
    for n in range(7):
        p1 = get_point(client, h1["id"], n)
        p2 = get_point(client, h2["id"], n)
        assert p1["version"] == 1 and p2["version"] == 2
        assert p1["burnup_id"] == h1["id"] and p2["burnup_id"] == h2["id"]
    # 初始成分相同 → 点 0 的 k 相同；燃耗参数不同 → 末态 k 明显不同
    assert abs(f1["current_k"] - f2["current_k"]) > 1e-4

    # 另起一条显式绑版本 1 的历程：与 h1 逐点一致（h1 认的是当初那一版）
    h3 = start_burnup(client, cid, [10.0] * 6, version=1)
    f3 = wait_burnup(client, h3["id"])
    assert f3["version"] == 1 and f3["status"] == "completed"
    for n in range(7):
        p1 = get_point(client, h1["id"], n)
        p3 = get_point(client, h3["id"], n)
        assert abs(p1["k_eff"] - p3["k_eff"]) <= 1e-12
        s1, s3 = p1["region_remaining"][0], p3["region_remaining"][0]
        assert abs(s1 - s3) / s3 <= 1e-9


# ---------- 8. 输入拒收：逐项点出字段 ----------

def _post_case_raw(client, regions):
    return client.post("/cases", json=dict(name="校验", regions=regions))


def test_burnup_field_validation(client):
    # 初始数密度为负 / 为零
    for bad in (-1.0, 0.0):
        r = _post_case_raw(client, [burn_fuel(
            50.0, 10, burnup=dict(BURN, initial_density=bad))])
        assert r.status_code == 422
        assert "initial_density" in r.text
    # 微观截面为负
    r = _post_case_raw(client, [burn_fuel(
        50.0, 10, burnup=dict(BURN, micro_sigma_a=-1.0))])
    assert r.status_code == 422 and "micro_sigma_a" in r.text
    # 易裂变吸收份额不在 [0,1]
    for bad in (-0.1, 1.5):
        r = _post_case_raw(client, [burn_fuel(
            50.0, 10, burnup=dict(BURN, fissile_absorption_fraction=bad))])
        assert r.status_code == 422 and "fissile_absorption_fraction" in r.text
    # 在不含裂变材料的区声明燃耗参数
    regions = [bare_fuel(50.0, 10),
               dict(thickness=10.0, D=1.0, sigma_a=0.05, nu_sigma_f=0.0,
                    n_mesh=10, burnup=dict(BURN))]
    r = _post_case_raw(client, regions)
    assert r.status_code == 422
    assert "burnup" in r.text and "裂变" in r.text


def test_burnup_steps_validation(client):
    case = make_burn_case(client, [burn_fuel(50.0, 10)])
    cid = case["id"]
    # 时间步长度不大于零
    for bad_steps in ([0.0], [-5.0], [10.0, 0.0]):
        r = client.post(f"/cases/{cid}/burnups", json={"steps": bad_steps})
        assert r.status_code == 422 and "steps" in r.text
    # 步数超过 500
    r = client.post(f"/cases/{cid}/burnups", json={"steps": [1.0] * 501})
    assert r.status_code == 422 and "steps" in r.text
    # 绑定的版本不存在
    r = client.post(f"/cases/{cid}/burnups",
                    json={"steps": [1.0], "version": 99})
    assert r.status_code == 404


def test_burnup_patch_validation(client):
    regions = [bare_fuel(50.0, 10),
               dict(thickness=10.0, D=1.0, sigma_a=0.05, nu_sigma_f=0.0,
                    n_mesh=10)]
    case = make_case(client, regions)
    cid = case["id"]
    # 给非裂变区补声明燃耗 → 拒绝，不产生新版本
    r = client.patch(f"/cases/{cid}", json={
        "region_index": 1, "region": {"burnup": dict(BURN)}})
    assert r.status_code == 422 and "burnup" in r.text
    assert client.get(f"/cases/{cid}").json()["current_version"] == 1
    # 非法份额的补丁 → 拒绝
    r = client.patch(f"/cases/{cid}", json={
        "region_index": 0,
        "region": {"burnup": dict(BURN, fissile_absorption_fraction=2.0)}})
    assert r.status_code == 422
    assert client.get(f"/cases/{cid}").json()["current_version"] == 1
    # 合法补丁：给燃料区加燃耗声明 → 新版本
    r = client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"burnup": dict(BURN)}})
    assert r.status_code == 200, r.text
    assert r.json()["current_version"] == 2
    assert (r.json()["regions"][0]["burnup"]["initial_density"]
            == BURN["initial_density"])


# ---------- 9. 老数据兼容：无燃耗声明的工况/结果原样可读、重解一致 ----------

def test_legacy_case_without_burnup_readable_and_reproducible(data_dir):
    # 手写一份"改造前"格式的工况 JSON（regions 没有 burnup 键）
    legacy = {
        "id": "case_legacy", "name": "老工况",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "current_version": 1,
        "versions": [{
            "version": 1, "created_at": "2026-01-01T00:00:00+00:00",
            "change_summary": "初始版本",
            "regions": [dict(thickness=50.0, D=1.0, sigma_a=0.1,
                             nu_sigma_f=0.12, n_mesh=100)],
            "left_boundary": "zero_flux", "right_boundary": "zero_flux",
            "settings": {"k_tol": 1e-9, "flux_tol": 1e-9, "max_iter": 100000},
            "target_total_fission": 1.0, "result_ids": [],
        }],
    }
    os.makedirs(os.path.join(data_dir, "cases"), exist_ok=True)
    with open(os.path.join(data_dir, "cases", "case_legacy.json"),
              "w", encoding="utf-8") as fh:
        json.dump(legacy, fh, ensure_ascii=False)

    storage = Storage(data_dir)  # 模拟升级后重启：老数据原样可读
    app = create_app(storage)
    with TestClient(app) as c:
        got = c.get("/cases/case_legacy").json()
        assert got["name"] == "老工况"
        assert got["regions"][0].get("burnup") is None
        versions = c.get("/cases/case_legacy/versions").json()
        assert "burnup" not in versions[0]["regions"][0]

        r = c.post("/cases/case_legacy/solve", json={"mode": "cold"})
        assert r.status_code == 200, r.text
        k_legacy = r.json()["k_eff"]
        # 同参数新工况重解：k 一致到收敛容差以内很多（同一数值内核）
        fresh = make_case(c, [bare_fuel(50.0, 100)])
        k_fresh = solve_case(c, fresh["id"], mode="cold")["k_eff"]
        assert abs(k_legacy - k_fresh) < 1e-12

        # 不声明燃耗的工况也能建历程：参数始终不变，k 不动
        h = start_burnup(c, "case_legacy", [10.0] * 3)
        f = wait_burnup(c, h["id"])
        assert f["status"] == "completed"
        for n in range(4):
            p = get_point(c, h["id"], n)
            assert p["region_remaining"] == [1.0]
            assert abs(p["k_eff"] - k_legacy) < 1e-9
    app.state.jobs.shutdown()
    app.state.burnups.shutdown()
