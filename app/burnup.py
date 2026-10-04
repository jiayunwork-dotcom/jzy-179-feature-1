"""燃耗引擎：空间分辨的易裂变核素消耗与预估-校正时间推进。

物理口径（需求固定，不在此处变通）：
- 燃料区可声明：易裂变核素初始数密度 N0、微观吸收截面 σa（barn）、
  该区 Σa 中来自易裂变核素的份额 f ∈ [0,1]；未声明的区不燃耗，参数不变。
- 消耗率正比于当地通量与当地剩余数密度：dN/dt = −σa·φ·N。
- νΣf 与 Σa 的易裂变部分按 N/N0 同步缩小；Σa 其余部分与 D 保持不变。
- 全程恒定总裂变率：每个时刻通量都归一到工况给定的总裂变率 F，
  因此燃料越烧、通量幅值越高。

实现要点：
- 状态是**逐单元**剩余份额 r_i = N_i/N0（同一区里不同位置消耗快慢不同，
  "一区一套截面"从第一步起就不再成立）。Mesh 本来就逐单元存 Σa/νΣf，
  因此离散、求解、逐区反应率统计都不需要改，只需按 r_i 重算逐单元截面：
      νΣf_i(t) = νΣf_i,0 · r_i
      Σa_i(t)  = Σa_i,0 · (1 − f) + Σa_i,0 · f · r_i
- 数密度以比值 r 跟踪即可：dN/dt = −σφN 对 N 是线性的，N0 在比值中
  约去（绝对数密度 N_i = N0·r_i 可随时还原）。声明中的 N0 参与校验并
  随快照存档，不进入递推——这是恒等化简，不是近似。
- 时间推进用**预估-校正（Heun 二阶）**，详见 README"燃耗时间推进"：
      预估  r^p   = rⁿ·exp(−σ̃·φⁿ·Δt)              （步首通量冻结积分）
      校正  rⁿ⁺¹  = rⁿ·exp(−σ̃·(φⁿ+φ^p)/2·Δt)      （梯形平均通量）
  其中 σ̃ = σa·1e-24·86400（barn→cm²、秒→天）。指数积分器严格保持
  r ∈ (0,1]，通量非负由求解器保证。全局误差 O(Δt²)：步长减半，误差约 1/4。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .discretization import BoundaryCondition, Mesh
from .solver import (
    SolveResult,
    SolverError,
    SolverSettings,
    solve,
)

SECONDS_PER_DAY = 86400.0
BARN_TO_CM2 = 1e-24


class FuelExhaustedError(RuntimeError):
    """易裂变核素耗尽：恒定总裂变率要求的通量趋于无穷，特征值问题失去意义。

    指数积分器严格保持剩余份额 > 0，但份额下溢到 0（裂变权重为 0）时，
    "归一到给定总裂变率"在数学上无解——这就是"烧不动"的终点。
    """


@dataclass(frozen=True)
class BurnupParams:
    """一个区的燃耗声明（已校验）。"""
    fissile_density_0: float              # N0 (10^24 cm^-3)
    sigma_a_micro: float                  # 微观吸收截面 (barn)
    fissile_absorption_fraction: float    # Σa 中易裂变份额 f ∈ [0,1]

    @property
    def depletion_coeff(self) -> float:
        """σ̃：单位通量下每天的相对消耗率（σa[barn] → cm²，秒 → 天）。"""
        return self.sigma_a_micro * BARN_TO_CM2 * SECONDS_PER_DAY


def parse_burnup_params(regions) -> list[Optional[BurnupParams]]:
    """从版本快照的区参数里解析燃耗声明；未声明的区为 None。"""
    params: list[Optional[BurnupParams]] = []
    for r in regions:
        burn = r.get("burnup") if isinstance(r, dict) else getattr(r, "burnup", None)
        if burn is None:
            params.append(None)
            continue
        params.append(BurnupParams(
            fissile_density_0=float(burn["fissile_density_0"]),
            sigma_a_micro=float(burn["sigma_a_micro"]),
            fissile_absorption_fraction=float(burn["fissile_absorption_fraction"]),
        ))
    return params


def cell_depletion_coeff(mesh: Mesh,
                         params: list[Optional[BurnupParams]]) -> np.ndarray:
    """逐单元 σ̃（不燃耗的单元为 0）。"""
    coeff = np.zeros(mesh.n, dtype=float)
    for r_idx, p in enumerate(params):
        if p is None:
            continue
        coeff[mesh.region_of == r_idx] = p.depletion_coeff
    return coeff


def depleted_mesh(mesh: Mesh, params: list[Optional[BurnupParams]],
                  remaining: np.ndarray) -> Mesh:
    """按逐单元剩余份额重算逐单元 νΣf 与 Σa（D 与其余 Σa 不动）。

    mesh.sigma_a / mesh.nu_sigma_f 存的是初始（t=0）逐单元值；
    remaining[i] = N_i/N0 ∈ (0,1]，不燃耗单元恒为 1。
    """
    if all(p is None for p in params):
        return mesh
    sigma_a = np.array(mesh.sigma_a, dtype=float, copy=True)
    nu_sigma_f = np.array(mesh.nu_sigma_f, dtype=float, copy=True)
    for r_idx, p in enumerate(params):
        if p is None:
            continue
        mask = mesh.region_of == r_idx
        r = remaining[mask]
        f = p.fissile_absorption_fraction
        nu_sigma_f[mask] = mesh.nu_sigma_f[mask] * r
        sigma_a[mask] = mesh.sigma_a[mask] * (1.0 - f) + \
            mesh.sigma_a[mask] * f * r
    return replace(mesh, sigma_a=sigma_a, nu_sigma_f=nu_sigma_f)


def region_remaining(mesh: Mesh, remaining: np.ndarray) -> list[float]:
    """各区剩余易裂变份额：区内剩余原子数 / 初始原子数（体积加权平均）。"""
    out: list[float] = []
    for r_idx in range(len(mesh.region_hi)):
        mask = mesh.region_of == r_idx
        w = mesh.widths[mask]
        out.append(float(np.dot(remaining[mask], w) / np.sum(w)))
    return out


def _fission_weight(mesh: Mesh, params: list[Optional[BurnupParams]],
                    remaining: np.ndarray) -> float:
    """耗尽检查：当前状态下的总裂变权重 Σ νΣf·h（归一化的分母）。"""
    dm = depleted_mesh(mesh, params, remaining)
    return float(np.sum(dm.nu_sigma_f * dm.widths))


def _check_not_exhausted(mesh: Mesh, params: list[Optional[BurnupParams]],
                         remaining: np.ndarray, stage: str) -> None:
    w = _fission_weight(mesh, params, remaining)
    if not np.isfinite(w) or w <= 0.0:
        raise FuelExhaustedError(
            f"{stage}：易裂变核素耗尽（裂变权重为 0），"
            "无法继续按给定总裂变率归一")


def _solve_hot_or_cold(mesh: Mesh, left: BoundaryCondition,
                       right: BoundaryCondition, settings: SolverSettings,
                       *, previous_phi: np.ndarray, initial_k: float,
                       target_total_fission: float) -> SolveResult:
    """优先热启动（燃耗步之间状态接近，Wielandt 几步即收敛）；
    热启动本身失败时退回冷启动，保证结果与起点无关。"""
    try:
        return solve(mesh, left, right, settings, mode="hot",
                     previous_phi=previous_phi, initial_k=initial_k,
                     target_total_fission=target_total_fission)
    except SolverError:
        return solve(mesh, left, right, settings, mode="cold",
                     target_total_fission=target_total_fission)


def depletion_step(
    mesh: Mesh,
    left: BoundaryCondition,
    right: BoundaryCondition,
    settings: SolverSettings,
    params: list[Optional[BurnupParams]],
    remaining: np.ndarray,
    phi: np.ndarray,
    k: float,
    step_days: float,
    target_total_fission: float,
) -> tuple[np.ndarray, SolveResult]:
    """推进一个燃耗步（预估-校正，Heun 二阶）。

    输入：步首状态（remaining, phi, k），phi 已按 target_total_fission 归一。
    输出：步末逐单元剩余份额与该点的完整求解结果（k、通量、反应率、平衡）。

    每步两次特征值求解：预估态一次、校正后的步末态一次（即下一时间点的解）。
    """
    coeff = cell_depletion_coeff(mesh, params)

    # --- 预估：步首通量冻结，指数积分到步末 ---
    r_pred = remaining * np.exp(-coeff * phi * step_days)
    _check_not_exhausted(mesh, params, r_pred, "预估态")
    mesh_pred = depleted_mesh(mesh, params, r_pred)
    res_pred = _solve_hot_or_cold(
        mesh_pred, left, right, settings,
        previous_phi=phi, initial_k=k,
        target_total_fission=target_total_fission)

    # --- 校正：用步首与预估态的梯形平均通量重新积分 ---
    phi_avg = 0.5 * (phi + res_pred.phi)
    r_new = remaining * np.exp(-coeff * phi_avg * step_days)
    _check_not_exhausted(mesh, params, r_new, "步末态")
    mesh_new = depleted_mesh(mesh, params, r_new)
    res_new = _solve_hot_or_cold(
        mesh_new, left, right, settings,
        previous_phi=res_pred.phi, initial_k=res_pred.k_eff,
        target_total_fission=target_total_fission)
    return r_new, res_new
