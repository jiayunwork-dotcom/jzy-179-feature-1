"""物理判据测试：裸板解析、加密二阶、对称/反射、反射层、单调性、D 效应、非负。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.discretization import (
    BoundaryCondition,
    Region,
    build_mesh,
)
from app.physics import analytical_k, k_infinity, l_squared
from app.solver import SolverSettings, solve

from conftest import REFLECTOR, bare_fuel


SETT = SolverSettings(k_tol=1e-11, flux_tol=1e-11, max_iter=200_000)


def k_of(reg_dict, left="zero_flux", right="zero_flux", settings=SETT):
    mesh = build_mesh([Region(**reg_dict)])
    return solve(mesh, BoundaryCondition(left), BoundaryCondition(right), settings)


# ---------- 裸板解析 k∞/(1+L²B²) ----------

def test_bare_slab_k_matches_analytic_zero_flux():
    reg = Region(**bare_fuel(50.0, 400))
    res = k_of(bare_fuel(50.0, 400))
    k_an = analytical_k(reg, "zero_flux", "zero_flux")
    assert abs(res.k_eff - k_an) / k_an < 2e-7
    # 直接核对 k∞、L²
    assert math.isclose(k_infinity(reg), 1.2, rel_tol=1e-14)
    assert math.isclose(l_squared(reg), 10.0, rel_tol=1e-14)


def test_bare_slab_k_matches_analytic_extrapolated():
    reg = Region(**bare_fuel(50.0, 400))
    res = k_of(bare_fuel(50.0, 400), "extrapolated", "extrapolated")
    k_an = analytical_k(reg, "extrapolated", "extrapolated")
    assert abs(res.k_eff - k_an) / k_an < 2e-7


@pytest.mark.parametrize("left,right", [
    ("zero_flux", "reflective"),
    ("extrapolated", "reflective"),
])
def test_bare_slab_k_matches_analytic_mixed(left, right):
    reg = Region(**bare_fuel(50.0, 400))
    res = k_of(bare_fuel(50.0, 400), left, right)
    k_an = analytical_k(reg, left, right)
    assert abs(res.k_eff - k_an) / k_an < 2e-7


# ---------- 网格逐级加密、误差 ~1/4、二阶收敛 ----------

@pytest.mark.parametrize("left,right", [
    ("zero_flux", "zero_flux"),
    ("extrapolated", "extrapolated"),
    ("zero_flux", "reflective"),
    ("extrapolated", "reflective"),
])
def test_mesh_refinement_order_is_two(left, right):
    reg = Region(**bare_fuel(50.0, 1))
    k_an = analytical_k(reg, left, right)
    Ns = [25, 50, 100, 200, 400]
    errs = []
    for N in Ns:
        r = k_of(bare_fuel(50.0, N), left, right)
        errs.append(abs(r.k_eff - k_an))
    # 每级误差比 ~4，拟合 log2(err) vs log2(h) 的斜率 ≈ 2
    hs = np.array([50.0 / N for N in Ns])
    log_h = np.log(hs)
    log_e = np.log(np.array(errs))
    # 用后三级（渐近区）拟合阶数
    slope, _ = np.polyfit(log_h[-3:], log_e[-3:], 1)
    assert 1.9 < slope < 2.1, f"{left}/{right} 收敛阶 {slope:.3f} 不是二阶"
    # 逐级误差比（除第一级外）都在 4 附近
    for e_prev, e_next in zip(errs[1:-1], errs[2:-1]):
        assert 3.7 < e_prev / e_next < 4.3


# ---------- 左右对称、峰在中心 ----------

def test_symmetric_problem_symmetric_flux():
    mesh = build_mesh([Region(20.0, 1.0, 0.0, 0.0, 80),
                       Region(30.0, 1.0, 0.1, 0.12, 120),
                       Region(20.0, 1.0, 0.0, 0.0, 80)])
    res = solve(mesh, BoundaryCondition("zero_flux"),
                BoundaryCondition("zero_flux"), SETT)
    phi = res.phi
    assert np.max(np.abs(phi - phi[::-1])) / phi.max() < 1e-10
    # 峰在整板几何中心附近（一个网格宽度内）
    centers = res.centers
    assert abs(centers[int(np.argmax(phi))] - 35.0) <= 35.0 / 280 + 1e-9
    assert np.all(phi >= 0.0)


# ---------- 半块板 + 中心全反射 == 整块对称板 ----------

def test_reflective_half_equals_full_slab():
    # 整板 a=50、200 份；半板 25、100 份，外端零通量、内端全反射
    full = k_of(bare_fuel(50.0, 200), "zero_flux", "zero_flux")
    half = k_of(bare_fuel(25.0, 100), "zero_flux", "reflective")
    assert abs(full.k_eff - half.k_eff) < 1e-10

    # 通量形状：半板应与整板左半一致（半板总裂变率只有整块一半，
    # 都按总裂变率=1 归一，故半板幅度恰为整块对应部分的 2 倍）
    np.testing.assert_allclose(half.phi, 2.0 * full.phi[:100], rtol=1e-10, atol=1e-12)

    # 同样对"外推边界"做一次
    full_e = k_of(bare_fuel(50.0, 200), "extrapolated", "extrapolated")
    half_e = k_of(bare_fuel(25.0, 100), "extrapolated", "reflective")
    assert abs(full_e.k_eff - half_e.k_eff) < 1e-8


# ---------- 反射层使 k 变大 ----------

def test_reflector_increases_k():
    bare = k_of(bare_fuel(30.0, 120), "zero_flux", "zero_flux")
    mesh = build_mesh([Region(**REFLECTOR),
                       Region(thickness=30.0, D=1.0, sigma_a=0.1,
                              nu_sigma_f=0.12, n_mesh=120),
                       Region(**REFLECTOR)])
    reflected = solve(mesh, BoundaryCondition("zero_flux"),
                      BoundaryCondition("zero_flux"), SETT)
    assert reflected.k_eff > bare.k_eff
    # 反射层只散射：其吸收率严格为 0，但有泄漏返回效果
    assert reflected.region_absorption[0] == 0.0
    assert reflected.region_absorption[2] == 0.0


# ---------- k∞>1 时增厚燃料 k 单调上升 ----------

def test_k_monotonic_in_fuel_thickness():
    ks = [k_of(bare_fuel(a, max(a * 4, 40))).k_eff
          for a in (10.0, 20.0, 30.0, 40.0, 60.0, 80.0)]
    for k_prev, k_next in zip(ks, ks[1:]):
        assert k_next > k_prev
    # 厚板极限从下方趋近 k∞
    assert ks[-1] < 1.2
    assert abs(ks[-1] - 1.2) < 0.03


# ---------- 只调大 D，裸板 k 变小 ----------

def test_k_decreases_with_D():
    ks = [k_of(bare_fuel(50.0, 200, D=D)).k_eff
          for D in (0.5, 1.0, 2.0, 4.0, 8.0)]
    for k_prev, k_next in zip(ks, ks[1:]):
        assert k_next < k_prev


# ---------- 任何网格点通量非负（含纯散射反射层） ----------

def test_flux_nonnegative_everywhere():
    mesh = build_mesh([Region(**REFLECTOR),
                       Region(thickness=30.0, D=1.0, sigma_a=0.1,
                              nu_sigma_f=0.12, n_mesh=120),
                       Region(**REFLECTOR)])
    res = solve(mesh, BoundaryCondition("extrapolated"),
                BoundaryCondition("extrapolated"), SETT)
    assert np.all(np.isfinite(res.phi))
    assert np.all(res.phi >= 0.0)
    # 反射层里通量应为正（泄漏进去的）
    assert res.phi[:80].min() > 0.0


# ---------- 界面连续性（不平均截面）：构造 D、Σa 跳变检查 J 连续 ----------

def test_interface_current_continuity_with_coefficient_jump():
    # 燃料 D=2, sa=0.1 | 反射层 D=0.5, sa=0；界面流必须严格同一个值
    fuel = Region(thickness=30.0, D=2.0, sigma_a=0.1, nu_sigma_f=0.12, n_mesh=120)
    refl = Region(thickness=10.0, D=0.5, sigma_a=0.0, nu_sigma_f=0.0, n_mesh=40)
    mesh = build_mesh([fuel, refl])
    res = solve(mesh, BoundaryCondition("zero_flux"),
                BoundaryCondition("reflective"), SETT)
    phi = res.phi
    # 界面是单元 119（燃料末）与 120（反射层首）之间的内面
    i = 119
    D_l, h_l = mesh.D[i], mesh.widths[i]
    D_r, h_r = mesh.D[i + 1], mesh.widths[i + 1]
    g = 2.0 * D_l * D_r / (D_l * h_r + D_r * h_l)
    j_from_left = g * (phi[i] - phi[i + 1])
    # 有限体积中两侧平衡共用这一个流：直接验证单元 119、120 的平衡残差为机器零
    l, d, u, f = None, None, None, None
    from app.discretization import assemble_operators
    lower, diag, upper, fiss = assemble_operators(
        mesh, BoundaryCondition("zero_flux"),
        BoundaryCondition("reflective"))

    def row_residual(idx):
        mphi = (lower[idx] * phi[idx - 1] if idx > 0 else 0.0) \
            + diag[idx] * phi[idx] \
            + (upper[idx] * phi[idx + 1] if idx + 1 < phi.size else 0.0)
        return abs(mphi - fiss[idx] * phi[idx] / res.k_eff)

    assert row_residual(i) < 1e-10
    assert row_residual(i + 1) < 1e-10
    assert j_from_left != 0.0
