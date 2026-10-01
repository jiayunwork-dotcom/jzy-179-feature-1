"""中子平衡与逐区反应率、两端泄漏率。"""
from __future__ import annotations

import numpy as np

from app.discretization import BoundaryCondition, Region, build_mesh
from app.solver import SolverSettings, solve

from conftest import REFLECTOR

SETT = SolverSettings(k_tol=1e-11, flux_tol=1e-11, max_iter=200_000)


def _check_balance(res):
    production = res.total_fission / res.k_eff
    loss = res.total_absorption + res.leakage_left + res.leakage_right
    rel = abs(production - loss) / production
    assert rel < 1e-8, f"中子平衡残差 {rel:.3e} 超 1e-8"
    return rel


def test_balance_bare_slab():
    mesh = build_mesh([Region(50.0, 1.0, 0.1, 0.12, 200)])
    res = solve(mesh, BoundaryCondition("zero_flux"),
                BoundaryCondition("zero_flux"), SETT)
    _check_balance(res)
    # 对称问题两端泄漏相等且为正
    assert res.leakage_left > 0 and res.leakage_right > 0
    assert abs(res.leakage_left - res.leakage_right) / res.leakage_left < 1e-10


def test_balance_reflective_end_zero_leak():
    mesh = build_mesh([Region(25.0, 1.0, 0.1, 0.12, 100)])
    res = solve(mesh, BoundaryCondition("zero_flux"),
                BoundaryCondition("reflective"), SETT)
    _check_balance(res)
    assert res.leakage_right == 0.0
    assert res.leakage_left > 0.0


def test_balance_multizone_reflector():
    mesh = build_mesh([Region(**REFLECTOR),
                       Region(30.0, 1.0, 0.1, 0.12, 120),
                       Region(**REFLECTOR)])
    res = solve(mesh, BoundaryCondition("extrapolated"),
                BoundaryCondition("extrapolated"), SETT)
    _check_balance(res)
    # 逐区值与全场重算一致
    assert abs(sum(res.region_absorption) - res.total_absorption) < 1e-12
    assert abs(sum(res.region_fission) - res.total_fission) < 1e-12
    # 反射层无吸收、无裂变
    assert res.region_absorption[0] == 0.0
    assert res.region_absorption[2] == 0.0
    assert res.region_fission[0] == 0.0
    assert res.region_fission[2] == 0.0
    assert res.region_fission[1] > 0.0


def test_normalization_to_target_fission():
    mesh = build_mesh([Region(50.0, 1.0, 0.1, 0.12, 200)])
    for target in (1.0, 5.0, 1.0e6):
        res = solve(mesh, BoundaryCondition("zero_flux"),
                    BoundaryCondition("zero_flux"), SETT,
                    target_total_fission=target)
        assert abs(res.total_fission - target) / target < 1e-12
        _check_balance(res)


def test_cellwise_balance_sums_to_global():
    """逐单元有限体积方程加起来就是全局平衡（望远镜求和），残差应在机器精度。"""
    mesh = build_mesh([Region(10.0, 2.0, 0.05, 0.08, 40),
                       Region(20.0, 1.0, 0.1, 0.12, 80),
                       Region(8.0, 0.5, 0.0, 0.0, 32)])
    res = solve(mesh, BoundaryCondition("zero_flux"),
                BoundaryCondition("zero_flux"), SETT)
    assert res.balance_residual < 1e-10
    _check_balance(res)
