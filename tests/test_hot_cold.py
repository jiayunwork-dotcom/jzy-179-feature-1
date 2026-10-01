"""冷/热启动一致性与步数对比。

同一次改动（改某一区参数，网格数也变）：
- 冷启动：平坦初值
- 热启动：上一版收敛通量按新网格分段线性重映 + 继承 k，走 Wielandt 加速
要求两条路径的 k 差落在收敛容差量级、归一化通量形状一致，
且中子平衡两条路径都过 1e-8。同时打印步数对比（交付说明引用该输出）。
"""
from __future__ import annotations

import numpy as np

from app.discretization import BoundaryCondition, build_mesh
from app.interpolation import piecewise_linear_remap
from app.solver import SolverSettings, solve

from conftest import bare_fuel, client, make_case, solve_case  # noqa: F401

SETT = SolverSettings(k_tol=1e-9, flux_tol=1e-9, max_iter=100_000)


def _direct_cold_hot(reg1, reg2):
    m1 = build_mesh([__import__("app.discretization", fromlist=["Region"]).Region(**reg1)])
    m2 = build_mesh([__import__("app.discretization", fromlist=["Region"]).Region(**reg2)])
    bc = (BoundaryCondition("zero_flux"), BoundaryCondition("zero_flux"))
    r1 = solve(m1, *bc, SETT)
    rc = solve(m2, *bc, SETT, mode="cold")
    guess = piecewise_linear_remap(m1.centers, r1.phi, m2.centers)
    rh = solve(m2, *bc, SETT, mode="hot", previous_phi=guess,
               initial_k=r1.k_eff)
    return r1, rc, rh


def test_hot_matches_cold_small_thickness_change():
    r1, rc, rh = _direct_cold_hot(
        bare_fuel(50.0, 100), bare_fuel(52.0, 120))
    dk = abs(rc.k_eff - rh.k_eff) / rc.k_eff
    dphi = np.linalg.norm(rc.phi - rh.phi) / np.linalg.norm(rc.phi)
    assert dk < 1e-7
    assert dphi < 1e-6
    assert rc.balance_residual < 1e-8 and rh.balance_residual < 1e-8
    print(f"\n[冷热对比] 厚度 50→52 cm、网格 100→120："
          f"冷 {rc.iterations} 步，热 {rh.iterations} 步")
    # 热启动在"小改动"上必须明显省步
    assert rh.iterations < rc.iterations


def test_hot_matches_cold_mesh_only_change():
    # 参数完全相同、只加密网格：热启动应近乎一步到位
    r1, rc, rh = _direct_cold_hot(
        bare_fuel(50.0, 50), bare_fuel(50.0, 100))
    dk = abs(rc.k_eff - rh.k_eff) / rc.k_eff
    dphi = np.linalg.norm(rc.phi - rh.phi) / np.linalg.norm(rc.phi)
    assert dk < 1e-7
    assert dphi < 1e-6
    print(f"[冷热对比] 仅网格 50→100：冷 {rc.iterations} 步，热 {rh.iterations} 步")
    assert rh.iterations <= max(3, rc.iterations // 4)


def test_hot_matches_cold_cross_section_change():
    r1, rc, rh = _direct_cold_hot(
        bare_fuel(50.0, 100, nsf=0.12), bare_fuel(50.0, 100, nsf=0.125))
    dk = abs(rc.k_eff - rh.k_eff) / rc.k_eff
    dphi = np.linalg.norm(rc.phi - rh.phi) / np.linalg.norm(rc.phi)
    assert dk < 1e-7
    assert dphi < 1e-6
    print(f"[冷热对比] νΣf 0.120→0.125：冷 {rc.iterations} 步，热 {rh.iterations} 步")
    assert rh.iterations < rc.iterations


def test_hot_robust_under_large_change():
    # 改动很大（新 k 越过旧移位点）：允许退回普通迭代，但结果仍须与冷启动一致
    r1, rc, rh = _direct_cold_hot(
        bare_fuel(50.0, 100), bare_fuel(100.0, 200))
    dk = abs(rc.k_eff - rh.k_eff) / rc.k_eff
    dphi = np.linalg.norm(rc.phi - rh.phi) / np.linalg.norm(rc.phi)
    assert dk < 1e-7
    assert dphi < 1e-6
    assert rh.balance_residual < 1e-8
    print(f"[冷热对比] 大改 厚度 50→100：冷 {rc.iterations} 步，热 {rh.iterations} 步")
    assert rh.iterations <= rc.iterations


# ---------- 通过 HTTP 端到端：改区参数后冷、热各算一遍 ----------

def test_api_cold_hot_after_region_patch(client):
    case = make_case(client, [bare_fuel(50.0, 100)], name="冷热")
    cid = case["id"]
    v1_cold = solve_case(client, cid, mode="cold")
    assert v1_cold["version"] == 1

    # 改第 0 区（厚度与网格数都变），生成版本 2
    r = client.patch(f"/cases/{cid}", json={
        "region_index": 0,
        "region": {"thickness": 52.0, "n_mesh": 120},
    })
    assert r.status_code == 200, r.text
    assert r.json()["current_version"] == 2

    v2_cold = solve_case(client, cid, mode="cold")
    v2_hot = solve_case(client, cid, mode="hot")
    assert v2_cold["version"] == v2_hot["version"] == 2
    assert v2_hot["warm_from_result"] == v1_cold["id"]

    assert abs(v2_cold["k_eff"] - v2_hot["k_eff"]) / v2_cold["k_eff"] < 1e-7
    pc = np.array(v2_cold["phi"])
    ph = np.array(v2_hot["phi"])
    assert np.linalg.norm(pc - ph) / np.linalg.norm(pc) < 1e-6
    assert v2_cold["balance_residual"] < 1e-8
    assert v2_hot["balance_residual"] < 1e-8
    assert v2_hot["iterations"] < v2_cold["iterations"]

    # 旧版本结果不被覆盖
    results_v1 = client.get(f"/cases/{cid}/results", params={"version": 1}).json()
    results_v2 = client.get(f"/cases/{cid}/results", params={"version": 2}).json()
    assert len(results_v1) == 1 and len(results_v2) == 2
    assert results_v1[0]["id"] == v1_cold["id"]


def test_hot_without_prior_result_is_409(client):
    case = make_case(client, [bare_fuel(50.0, 100)])
    r = client.post(f"/cases/{case['id']}/solve", json={"mode": "hot"})
    assert r.status_code == 409
