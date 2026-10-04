"""燃耗历程：手推对照、零截面不变、单调性、对称性、空间分辨消耗、
分段/重启一致、取消、并发隔离、版本锁定、输入拒收、老数据兼容。
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.physics import analytical_k
from app.discretization import Region
from app.storage import Storage, utc_now

from conftest import bare_fuel, make_case, solve_case

# 标准燃耗声明：N0=1e-3 (10^24/cm³)，σa=500 barn，Σa 的 80% 来自易裂变核素
BURN = dict(fissile_density_0=1.0e-3, sigma_a_micro=500.0,
            fissile_absorption_fraction=0.8)
# 恒定总裂变率（满功率）：让通量 ~1e15，燃耗在"天"的量级上可见
POWER = 5.0e15


def fuel_burn(a=50.0, n=100, *, sa=0.1, nsf=0.12, burn=BURN, D=1.0):
    reg = dict(thickness=float(a), D=float(D), sigma_a=float(sa),
               nu_sigma_f=float(nsf), n_mesh=int(n))
    if burn is not None:
        reg["burnup"] = dict(burn)
    return reg


def make_burnup_case(client, regions, *, name="燃耗", left="reflective",
                     right="reflective", target=POWER, settings=None):
    body = dict(name=name, regions=regions, left_boundary=left,
                right_boundary=right, target_total_fission=target)
    if settings is not None:
        body["settings"] = settings
    r = client.post("/cases", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def submit_burnup(client, case_id, steps, **kw):
    r = client.post(f"/cases/{case_id}/burnups",
                    json=dict(steps=steps, **kw))
    assert r.status_code == 202, r.text
    return r.json()


def wait_burnup(client, burnup_id, timeout=60.0, step=0.02):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/burnups/{burnup_id}").json()
        if last["status"] not in ("queued", "running"):
            return last
        time.sleep(step)
    raise AssertionError(f"燃耗历程 {burnup_id} 超时未结束: {last}")


def get_points(client, burnup_id):
    return client.get(f"/burnups/{burnup_id}/points").json()


def get_point(client, burnup_id, index):
    r = client.get(f"/burnups/{burnup_id}/points/{index}")
    assert r.status_code == 200, r.text
    return r.json()


def hand_reflective(t, *, sigma_micro, F, nsf0, sa0, frac, a):
    """两端全反射单区（B=0）的手推解：
    φ 平 → dr/dt = -σ̃F/(νΣf0·a) 恒定 → r(t)=1-λt；k(t)=νΣf(r)/Σa(r)。
    """
    lam = sigma_micro * 1e-24 * 86400.0 * F / (nsf0 * a)
    r = 1.0 - lam * t
    k = nsf0 * r / (sa0 * (1.0 - frac) + sa0 * frac * r)
    phi = F / (nsf0 * r * a)
    return r, k, phi


# ---------- 1. 两端全反射单区：每个时间点贴住手推值，步长减半误差缩小 ----------

def test_reflective_slab_tracks_hand_derived(client):
    case = make_burnup_case(client, [fuel_burn(50.0, 100)])
    burn = submit_burnup(client, case["id"], [0.5] * 20)   # 20 步 × 0.5 天
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed", final
    assert final["steps_completed"] == 20
    assert final["n_points"] == 21
    assert abs(final["burnup_time_days"] - 10.0) < 1e-12

    args = dict(sigma_micro=BURN["sigma_a_micro"], F=POWER, nsf0=0.12,
                sa0=0.1, frac=BURN["fissile_absorption_fraction"], a=50.0)
    for i in range(21):
        p = get_point(client, burn["id"], i)
        t = p["time_days"]
        assert abs(t - 0.5 * i) < 1e-12
        r_hand, k_hand, phi_hand = hand_reflective(t, **args)
        assert abs(p["k_eff"] - k_hand) < 2e-5, \
            f"t={t}: k={p['k_eff']:.10f} 手推 {k_hand:.10f}"
        assert abs(p["region_remaining"][0] - r_hand) < 1e-4
        # 通量平且幅值贴手推（恒定总裂变率 → 越烧幅值越高）
        phi = np.asarray(p["phi"])
        assert phi.min() > 0.0
        assert abs(phi.max() - phi_hand) / phi_hand < 1e-3
        assert (phi.max() - phi.min()) / phi.max() < 1e-6
        assert p["balance_residual"] < 1e-8


def test_error_shrinks_quadratically_when_step_halved(client):
    """二阶格式：步长减半，末端 k 对手推值的偏差约变 1/4。"""
    case = make_burnup_case(client, [fuel_burn(50.0, 100)], name="减半")
    args = dict(sigma_micro=BURN["sigma_a_micro"], F=POWER, nsf0=0.12,
                sa0=0.1, frac=BURN["fissile_absorption_fraction"], a=50.0)
    _, k_hand_10, _ = hand_reflective(10.0, **args)

    errs = []
    for dt, n in ((1.0, 10), (0.5, 20)):
        burn = submit_burnup(client, case["id"], [dt] * n)
        final = wait_burnup(client, burn["id"])
        assert final["status"] == "completed"
        k_end = get_point(client, burn["id"], n)["k_eff"]
        errs.append(abs(k_end - k_hand_10))
    assert errs[0] > 0.0
    ratio = errs[0] / errs[1]
    assert 2.5 < ratio < 6.0, f"误差比 {ratio:.2f} 不在二阶格式预期的 4 附近"
    assert errs[1] < 2e-5


# ---------- 2. 微观吸收截面为零：所有时间点与燃耗前一致 ----------

def test_zero_micro_cross_section_freezes_state(client):
    burn0 = dict(BURN, sigma_a_micro=0.0)
    case = make_burnup_case(client, [fuel_burn(50.0, 60, burn=burn0)],
                            left="zero_flux", right="zero_flux")
    burn = submit_burnup(client, case["id"], [1.0] * 4)
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"

    p0 = get_point(client, burn["id"], 0)
    phi0 = np.asarray(p0["phi"])
    for i in range(1, 5):
        p = get_point(client, burn["id"], i)
        assert abs(p["k_eff"] - p0["k_eff"]) < 1e-8
        phi = np.asarray(p["phi"])
        assert np.linalg.norm(phi - phi0) / np.linalg.norm(phi0) < 1e-7
        # 份额严格停在 1（体积加权的舍入在 1e-12 内）
        assert abs(p["region_remaining"][0] - 1.0) < 1e-12
        assert all(r == 1.0 for r in p["cell_remaining"])


# ---------- 3. 只有消耗没有增殖：k 单调下降、份额在 (0,1]、通量非负且幅值上升 ----------

def test_k_decreases_monotonically_and_state_bounded(client):
    case = make_burnup_case(client, [fuel_burn(50.0, 80)],
                            left="zero_flux", right="zero_flux")
    burn = submit_burnup(client, case["id"], [2.0] * 8)
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"

    points = [get_point(client, burn["id"], i) for i in range(9)]
    ks = [p["k_eff"] for p in points]
    for k_prev, k_next in zip(ks, ks[1:]):
        assert k_next < k_prev, "只有消耗没有增殖，k 必须单调下降"
    phi_max = []
    for p in points:
        r = p["region_remaining"][0]
        assert 0.0 < r <= 1.0
        phi = np.asarray(p["phi"])
        assert np.all(phi >= 0.0)
        assert p["balance_residual"] < 1e-8
        phi_max.append(phi.max())
    # 恒定总裂变率：燃料越烧，通量幅值越高
    for a_, b_ in zip(phi_max, phi_max[1:]):
        assert b_ > a_
    # 剩余份额也单调下降
    rs = [p["region_remaining"][0] for p in points]
    for r_prev, r_next in zip(rs, rs[1:]):
        assert r_next < r_prev


# ---------- 4. 左右对称问题燃耗后依旧对称；未声明的区不燃耗 ----------

def test_burnup_preserves_left_right_symmetry(client):
    refl = dict(thickness=20.0, D=1.0, sigma_a=0.0, nu_sigma_f=0.0, n_mesh=40)
    case = make_burnup_case(
        client, [refl, fuel_burn(30.0, 60), dict(refl)],
        left="zero_flux", right="zero_flux", name="对称")
    burn = submit_burnup(client, case["id"], [1.0] * 4)
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"

    for i in range(5):
        p = get_point(client, burn["id"], i)
        phi = np.asarray(p["phi"])
        assert np.max(np.abs(phi - phi[::-1])) / phi.max() < 1e-9
        cell_r = np.asarray(p["cell_remaining"])
        assert np.max(np.abs(cell_r - cell_r[::-1])) < 1e-9
        # 两个反射层未声明燃耗：份额恒为 1，参数不变
        assert p["region_remaining"][0] == 1.0
        assert p["region_remaining"][2] == 1.0
        assert p["region_remaining"][1] <= 1.0


# ---------- 5. 空间分辨：同一区内中心烧得快、边缘烧得慢 ----------

def test_spatial_depletion_center_burns_faster(client):
    case = make_burnup_case(client, [fuel_burn(50.0, 100)],
                            left="zero_flux", right="zero_flux", name="空间")
    burn = submit_burnup(client, case["id"], [1.0] * 5)
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"

    p = get_point(client, burn["id"], 5)
    cell_r = np.asarray(p["cell_remaining"])
    assert cell_r.size == 100
    # 中心（通量高）烧得比边缘（通量低）快："一区一套截面"从第一步起就不成立
    center = cell_r[45:55].mean()
    edge = np.concatenate([cell_r[:5], cell_r[-5:]]).mean()
    assert center < edge < 1.0
    assert center < p["region_remaining"][0] < edge
    # 逐单元剩余份额与通量形状负相关：通量峰处份额最低
    assert int(np.argmin(cell_r)) == int(np.argmax(np.asarray(p["phi"])))


# ---------- 6. 分段推进（中间重启一次服务）与一次推到底逐点一致 ----------

def _shutdown_app(app):
    app.state.jobs.shutdown()
    app.state.burnups.shutdown()


def test_segmented_with_restart_matches_one_shot(data_dir):
    storage1 = Storage(data_dir)
    app1 = create_app(storage1)
    with TestClient(app1) as c1:
        case = make_burnup_case(c1, [fuel_burn(40.0, 60)], name="分段")
        cid = case["id"]
        # 历程 A：一次推 10 步
        burn_a = submit_burnup(c1, cid, [1.0] * 10)
        # 历程 B：先推 4 步
        burn_b = submit_burnup(c1, cid, [1.0] * 4)
        fa = wait_burnup(c1, burn_a["id"])
        fb = wait_burnup(c1, burn_b["id"])
        assert fa["status"] == fb["status"] == "completed"
        assert fa["n_points"] == 11 and fb["n_points"] == 5
    _shutdown_app(app1)

    # —— 模拟服务重启：同一数据目录新建 Storage 与应用 ——
    storage2 = Storage(data_dir)
    app2 = create_app(storage2)
    with TestClient(app2) as c2:
        got = c2.get(f"/burnups/{burn_b['id']}").json()
        assert got["status"] == "completed" and got["n_points"] == 5
        # 历程 B 再推 6 步
        r = c2.post(f"/burnups/{burn_b['id']}/advance",
                    json={"steps": [1.0] * 6})
        assert r.status_code == 202, r.text
        fb2 = wait_burnup(c2, burn_b["id"])
        assert fb2["status"] == "completed"
        assert fb2["steps_completed"] == 10 and fb2["n_points"] == 11

        for i in range(11):
            pa = get_point(c2, burn_a["id"], i)
            pb = get_point(c2, burn_b["id"], i)
            assert pa["time_days"] == pb["time_days"]
            # k 的差落在收敛容差量级（默认 k_tol=1e-9）
            assert abs(pa["k_eff"] - pb["k_eff"]) < 5e-9, \
                f"点 {i}: Δk={abs(pa['k_eff'] - pb['k_eff']):.3e}"
            # 各区剩余份额相对差 ≤ 1e-9
            for ra, rb in zip(pa["region_remaining"], pb["region_remaining"]):
                assert abs(ra - rb) / ra < 1e-9
    _shutdown_app(app2)


def test_interrupted_history_resumes_after_restart(data_dir):
    """推进中服务崩溃重启 → interrupted，已完成点保留，可续推且结果正确。

    崩溃用"直接把历程状态拨回 running"模拟（等价于进程被杀时 worker
    没来得及写终态）：磁盘上留有 5 个已完成点，状态停在 running。
    """
    storage1 = Storage(data_dir)
    app1 = create_app(storage1)
    with TestClient(app1) as c1:
        case = make_burnup_case(c1, [fuel_burn(40.0, 60)], name="中断")
        burn = submit_burnup(c1, case["id"], [1.0] * 4)
        done = wait_burnup(c1, burn["id"])
        assert done["status"] == "completed" and done["n_points"] == 5
        # 模拟崩溃：worker 消失，状态遗留在 running
        storage1.update_burnup(burn["id"], status="running",
                               message="模拟崩溃现场")
    _shutdown_app(app1)

    storage2 = Storage(data_dir)
    app2 = create_app(storage2)   # 启动恢复：running → interrupted
    with TestClient(app2) as c2:
        got = c2.get(f"/burnups/{burn['id']}").json()
        assert got["status"] == "interrupted"
        assert got["n_points"] == 5 and got["steps_completed"] == 4
        # 已完成点都在
        assert len(get_points(c2, burn["id"])) == 5
        # 续推剩余 6 步
        r = c2.post(f"/burnups/{burn['id']}/advance",
                    json={"steps": [1.0] * 6})
        assert r.status_code == 202, r.text
        final = wait_burnup(c2, burn["id"])
        assert final["status"] == "completed"
        assert final["steps_completed"] == 10 and final["n_points"] == 11
        # 与手推值对照（同一 0D 问题），续推的每个点都贴住
        args = dict(sigma_micro=BURN["sigma_a_micro"], F=POWER, nsf0=0.12,
                    sa0=0.1, frac=BURN["fissile_absorption_fraction"], a=40.0)
        for i in range(11):
            p = get_point(c2, burn["id"], i)
            _, k_hand, _ = hand_reflective(p["time_days"], **args)
            assert abs(p["k_eff"] - k_hand) < 2e-4
    _shutdown_app(app2)


# ---------- 7. 取消：已完成的点保留，未完成的点不写入 ----------

def test_cancel_keeps_completed_points(data_dir):
    storage = Storage(data_dir)
    app = create_app(storage, burnup_step_delay=0.3)
    with TestClient(app) as c:
        case = make_burnup_case(c, [fuel_burn(40.0, 60)], name="取消")
        burn = submit_burnup(c, case["id"], [1.0] * 10)
        bid = burn["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            d = c.get(f"/burnups/{bid}").json()
            if d["n_points"] >= 3:
                break
            time.sleep(0.02)
        assert d["n_points"] >= 3
        assert c.post(f"/burnups/{bid}/cancel").status_code == 200

        final = wait_burnup(c, bid, timeout=30)
        assert final["status"] == "cancelled"
        assert final["cancelled"] is True
        n_points = final["n_points"]
        steps_done = final["steps_completed"]
        assert n_points == steps_done + 1
        assert 3 <= n_points < 11
        # 取消后不再有新点写入
        time.sleep(0.6)
        again = c.get(f"/burnups/{bid}").json()
        assert again["n_points"] == n_points
        # 已完成的点可逐个取回；下一个点不存在
        for i in range(n_points):
            get_point(c, bid, i)
        assert c.get(f"/burnups/{bid}/points/{n_points}").status_code == 404

        # 取消后还能接着往后推
        r = c.post(f"/burnups/{bid}/advance", json={"steps": [1.0] * 3})
        assert r.status_code == 202, r.text
        final2 = wait_burnup(c, bid)
        assert final2["status"] == "completed"
        assert final2["steps_completed"] == steps_done + 3
        assert final2["n_points"] == n_points + 3
    _shutdown_app(app)


def test_advance_while_running_is_409(data_dir):
    storage = Storage(data_dir)
    app = create_app(storage, burnup_step_delay=0.5)
    with TestClient(app) as c:
        case = make_burnup_case(c, [fuel_burn(40.0, 60)], name="冲突")
        burn = submit_burnup(c, case["id"], [1.0] * 10)
        r = c.post(f"/burnups/{burn['id']}/advance", json={"steps": [1.0]})
        assert r.status_code == 409
        final = wait_burnup(c, burn["id"])
        assert final["status"] == "completed"
    _shutdown_app(app)


# ---------- 8. 两条历程挂在同一工况的不同版本上同时推进，互不混入 ----------

def test_two_histories_on_different_versions_concurrent(data_dir):
    storage = Storage(data_dir)
    app = create_app(storage, burnup_step_delay=0.05)
    with TestClient(app) as c:
        case = make_burnup_case(c, [fuel_burn(40.0, 60, nsf=0.12)], name="并发")
        cid = case["id"]
        # H1 显式挂版本 1
        h1 = submit_burnup(c, cid, [1.0] * 5, version=1)
        # 改参数 → 版本 2（νΣf 0.12 → 0.20）
        pr = c.patch(f"/cases/{cid}", json={
            "region_index": 0, "region": {"nu_sigma_f": 0.20}})
        assert pr.status_code == 200 and pr.json()["current_version"] == 2
        # H2 挂版本 2；两条同时推进
        h2 = submit_burnup(c, cid, [1.0] * 5, version=2)

        f1 = wait_burnup(c, h1["id"])
        f2 = wait_burnup(c, h2["id"])
        assert f1["status"] == f2["status"] == "completed"
        assert f1["version"] == 1 and f2["version"] == 2

        k1 = [get_point(c, h1["id"], i)["k_eff"] for i in range(6)]
        k2 = [get_point(c, h2["id"], i)["k_eff"] for i in range(6)]
        # 各自初始 k 与各自版本的独立求解一致（不互相混入）
        from app.discretization import BoundaryCondition, Region, build_mesh
        from app.solver import SolverSettings, solve
        for nsf, k_expect in ((0.12, k1[0]), (0.20, k2[0])):
            mesh = build_mesh([Region(40.0, 1.0, 0.1, nsf, 60)])
            k_ref = solve(mesh, BoundaryCondition("reflective"),
                          BoundaryCondition("reflective"),
                          SolverSettings()).k_eff
            assert abs(k_expect - k_ref) < 1e-9
        # 两条历程的轨迹明显不同
        assert abs(k1[0] - k2[0]) > 0.1
        # 每个点都挂在各自历程、各自版本下
        for hid, ver in ((h1["id"], 1), (h2["id"], 2)):
            for s in get_points(c, hid):
                assert s["burnup_id"] == hid
                p = get_point(c, hid, s["step_index"])
                assert p["version"] == ver and p["burnup_id"] == hid
    _shutdown_app(app)


# ---------- 9. 工况后来改了参数，已有历程照旧认当初绑定的版本 ----------

def test_history_bound_to_original_version(client):
    case = make_burnup_case(client, [fuel_burn(40.0, 60, nsf=0.12)],
                            name="版本锁")
    cid = case["id"]
    k_v1 = solve_case(client, cid, mode="cold")["k_eff"]

    burn = submit_burnup(client, cid, [1.0] * 3)   # 锁版本 1
    # 立刻改工况 → 版本 2
    pr = client.patch(f"/cases/{cid}", json={
        "region_index": 0, "region": {"nu_sigma_f": 0.20}})
    assert pr.status_code == 200 and pr.json()["current_version"] == 2

    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed" and final["version"] == 1
    p0 = get_point(client, burn["id"], 0)
    assert abs(p0["k_eff"] - k_v1) < 1e-9   # 仍用版本 1 的参数

    # 改完之后再续推，依然认版本 1
    r = client.post(f"/burnups/{burn['id']}/advance", json={"steps": [1.0]})
    assert r.status_code == 202
    final2 = wait_burnup(client, burn["id"])
    assert final2["status"] == "completed" and final2["version"] == 1
    k_v2 = solve_case(client, cid, mode="cold")["k_eff"]
    assert abs(get_point(client, burn["id"], 0)["k_eff"] - k_v2) > 0.05


# ---------- 10. 进度与时间点内容 ----------

def test_progress_and_point_contents(client):
    case = make_burnup_case(client, [fuel_burn(40.0, 60)], name="进度")
    burn = submit_burnup(client, case["id"], [1.0, 2.0, 3.0])
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"
    assert final["steps_completed"] == 3
    assert final["n_points"] == 4
    assert abs(final["burnup_time_days"] - 6.0) < 1e-12
    last = get_point(client, burn["id"], 3)
    assert abs(final["current_k"] - last["k_eff"]) < 1e-15

    summaries = get_points(client, burn["id"])
    assert [s["step_index"] for s in summaries] == [0, 1, 2, 3]
    assert [s["time_days"] for s in summaries] == [0.0, 1.0, 3.0, 6.0]

    required = {"k_eff", "phi", "region_remaining", "region_absorption",
                "region_fission", "leakage_left", "leakage_right",
                "balance_residual", "cell_remaining", "centers", "widths",
                "region_of", "time_days", "step_days"}
    for i in range(4):
        p = get_point(client, burn["id"], i)
        assert required <= set(p), f"点 {i} 缺字段: {required - set(p)}"
        assert p["balance_residual"] < 1e-8
        assert len(p["phi"]) == p["n_mesh_total"] == 60
        assert len(p["cell_remaining"]) == 60
        assert abs(p["total_fission"] - POWER) / POWER < 1e-9  # 恒定总裂变率
    # 初始点就是新装料：份额全 1
    p0 = get_point(client, burn["id"], 0)
    assert p0["step_days"] == 0.0
    assert p0["region_remaining"] == [1.0]
    assert all(r == 1.0 for r in p0["cell_remaining"])


# ---------- 11. 声明了燃耗的工况，普通求解仍是新装料截面；老格式数据原样可读 ----------

def test_case_with_burnup_solves_like_fresh(client):
    with_burn = make_burnup_case(client, [fuel_burn(50.0, 200)],
                                 left="zero_flux", right="zero_flux",
                                 name="带燃耗")
    without = make_case(client, [bare_fuel(50.0, 200)], name="不带")
    k1 = solve_case(client, with_burn["id"], mode="cold")["k_eff"]
    k2 = solve_case(client, without["id"], mode="cold")["k_eff"]
    assert abs(k1 - k2) < 1e-9
    # 工况读出带 burnup 声明
    got = client.get(f"/cases/{with_burn['id']}").json()
    assert got["regions"][0]["burnup"]["sigma_a_micro"] == 500.0
    got2 = client.get(f"/cases/{without['id']}").json()
    assert got2["regions"][0]["burnup"] is None


def test_old_format_case_and_result_readable(data_dir):
    """燃耗功能引入前格式的工况/结果 JSON：原样可读，重解 k 与解析值一致。"""
    now = utc_now()
    case_doc = {
        "id": "case_oldfmt", "name": "老格式",
        "created_at": now, "updated_at": now, "current_version": 1,
        "versions": [{
            "version": 1, "created_at": now, "change_summary": "初始版本",
            "regions": [{"thickness": 50.0, "D": 1.0, "sigma_a": 0.1,
                         "nu_sigma_f": 0.12, "n_mesh": 400}],
            "left_boundary": "zero_flux", "right_boundary": "zero_flux",
            "settings": {"k_tol": 1e-9, "flux_tol": 1e-9, "max_iter": 100000},
            "target_total_fission": 1.0,
            "result_ids": ["res_oldfmt"],
        }],
    }
    result_doc = {
        "id": "res_oldfmt", "case_id": "case_oldfmt", "case_name": "老格式",
        "version": 1, "created_at": now, "mode": "cold",
        "k_eff": 1.161888152336, "iterations": 70,
        "k_residual": 1e-10, "flux_residual": 1e-10,
        "target_total_fission": 1.0,
        "region_absorption": [0.8615], "region_fission": [1.0],
        "leakage_left": 0.0692, "leakage_right": 0.0692,
        "total_absorption": 0.8615, "total_fission": 1.0,
        "balance_residual": 1e-12, "n_mesh_total": 200,
        "centers": [0.125], "widths": [0.25], "region_of": [0],
        "phi": [1.0], "note": None, "warm_from_result": None,
    }
    import os
    os.makedirs(os.path.join(data_dir, "cases"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "results"), exist_ok=True)
    with open(os.path.join(data_dir, "cases", "case_oldfmt.json"), "w",
              encoding="utf-8") as fh:
        json.dump(case_doc, fh)
    with open(os.path.join(data_dir, "results", "res_oldfmt.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result_doc, fh)

    app = create_app(Storage(data_dir))
    with TestClient(app) as c:
        got = c.get("/cases/case_oldfmt").json()
        assert got["regions"][0]["burnup"] is None      # 老数据没有该键
        old_res = c.get("/results/res_oldfmt").json()
        assert old_res["k_eff"] == result_doc["k_eff"]  # 老结果原样可读
        # 重新求解：k 与改造前的解析参考一致到收敛容差
        k_new = solve_case(c, "case_oldfmt", mode="cold")["k_eff"]
        k_an = analytical_k(Region(50.0, 1.0, 0.1, 0.12, 400),
                            "zero_flux", "zero_flux")
        assert abs(k_new - k_an) / k_an < 2e-7
    _shutdown_app(app)


# ---------- 12. 输入拒收（逐项点出字段） ----------

def _err_fields(r):
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    if isinstance(detail, dict):
        return {d["field"] for d in detail["details"]}
    return {tuple(e["loc"]) for e in detail}


def _post_case(client, regions):
    return client.post("/cases", json=dict(
        name="校验", regions=regions,
        left_boundary="reflective", right_boundary="reflective"))


def test_reject_negative_fissile_density(client):
    reg = fuel_burn(50.0, 50, burn=dict(BURN, fissile_density_0=-1.0))
    fields = _err_fields(_post_case(client, [reg]))
    assert any("fissile_density_0" in str(f) for f in fields)


def test_reject_negative_sigma_a_micro(client):
    reg = fuel_burn(50.0, 50, burn=dict(BURN, sigma_a_micro=-0.5))
    fields = _err_fields(_post_case(client, [reg]))
    assert any("sigma_a_micro" in str(f) for f in fields)


@pytest.mark.parametrize("frac", [-0.1, 1.5])
def test_reject_fissile_fraction_out_of_range(client, frac):
    reg = fuel_burn(50.0, 50, burn=dict(BURN, fissile_absorption_fraction=frac))
    fields = _err_fields(_post_case(client, [reg]))
    assert any("fissile_absorption_fraction" in str(f) for f in fields)


def test_reject_burnup_on_non_fissile_region(client):
    reg = dict(thickness=20.0, D=1.0, sigma_a=0.05, nu_sigma_f=0.0,
               n_mesh=20, burnup=dict(BURN))
    r = _post_case(client, [fuel_burn(50.0, 50), reg])
    assert r.status_code == 422
    assert "burnup" in r.text and "裂变" in r.text


def test_reject_burnup_patch_on_non_fissile_region(client):
    case = make_case(client, [
        bare_fuel(50.0, 50),
        dict(thickness=20.0, D=1.0, sigma_a=0.0, nu_sigma_f=0.0, n_mesh=20),
    ])
    r = client.patch(f"/cases/{case['id']}", json={
        "region_index": 1, "region": {"burnup": dict(BURN)}})
    assert r.status_code == 422
    assert client.get(f"/cases/{case['id']}").json()["current_version"] == 1


def test_reject_missing_burnup_fields(client):
    reg = fuel_burn(50.0, 50)
    reg["burnup"] = {"fissile_density_0": 1e-3}   # 缺两个字段
    r = _post_case(client, [reg])
    assert r.status_code == 422
    assert "sigma_a_micro" in r.text and "fissile_absorption_fraction" in r.text


@pytest.mark.parametrize("steps", [[0.0], [-1.0], []])
def test_reject_nonpositive_or_empty_steps(client, steps):
    case = make_burnup_case(client, [fuel_burn(40.0, 40)])
    r = client.post(f"/cases/{case['id']}/burnups", json={"steps": steps})
    assert r.status_code == 422
    assert "steps" in r.text


def test_reject_more_than_500_steps(client):
    case = make_burnup_case(client, [fuel_burn(40.0, 40)])
    r = client.post(f"/cases/{case['id']}/burnups",
                    json={"steps": [1.0] * 501})
    assert r.status_code == 422
    assert "steps" in r.text


def test_reject_cumulative_steps_over_500(client):
    # 微小 σ_micro：500 步内不会烧尽，专测累计步数上限
    tiny = dict(BURN, sigma_a_micro=0.05)
    case = make_burnup_case(client, [fuel_burn(40.0, 40, burn=tiny)])
    burn = submit_burnup(client, case["id"], [1.0])
    final = wait_burnup(client, burn["id"])
    assert final["status"] == "completed"      # 已完成 1 步
    r = client.post(f"/burnups/{burn['id']}/advance",
                    json={"steps": [1.0] * 500})   # 1 + 500 > 500
    assert r.status_code == 422
    assert "steps" in r.text
    # 499 步则可以（累计恰好 500）
    r2 = client.post(f"/burnups/{burn['id']}/advance",
                     json={"steps": [1.0] * 499})
    assert r2.status_code == 202
    final2 = wait_burnup(client, burn["id"], timeout=180)
    assert final2["status"] == "completed"
    assert final2["steps_completed"] == 500


def test_fuel_exhaustion_ends_run_cleanly(client):
    """满功率烧到底：裂变权重归零前历程以 exhausted 终态停下，点全部有效。"""
    case = make_burnup_case(client, [fuel_burn(40.0, 60)], name="烧尽")
    burn = submit_burnup(client, case["id"], [1.0] * 50)
    final = wait_burnup(client, burn["id"], timeout=120)
    assert final["status"] == "exhausted", final
    assert "耗尽" in final["message"]
    assert 0 < final["steps_completed"] < 50
    assert final["n_points"] == final["steps_completed"] + 1
    # 已完成的点全部有效：k > 0 且单调下降，份额在 (0,1]，平衡过关
    ks = []
    for i in range(final["n_points"]):
        p = get_point(client, burn["id"], i)
        assert p["k_eff"] > 0.0
        assert 0.0 < p["region_remaining"][0] <= 1.0
        assert p["balance_residual"] < 1e-8
        assert np.all(np.asarray(p["phi"]) >= 0.0)
        ks.append(p["k_eff"])
    for k_prev, k_next in zip(ks, ks[1:]):
        assert k_next < k_prev
    assert ks[-1] < 0.5   # 确实烧到了深度次临界（"烧不动"）


def test_burnup_unknown_case_or_history_404(client):
    assert client.post("/cases/nope/burnups",
                       json={"steps": [1.0]}).status_code == 404
    assert client.get("/burnups/nope").status_code == 404
    assert client.post("/burnups/nope/advance",
                       json={"steps": [1.0]}).status_code == 404
    assert client.post("/burnups/nope/cancel").status_code == 404
