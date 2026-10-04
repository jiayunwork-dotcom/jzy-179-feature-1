"""燃耗物理：易裂变核素的消耗与截面随剩余份额的缩放。

物理口径（需求固定）：
- 易裂变核素消耗速率正比于当地通量与当地剩余数密度：
      dN_i/dt = −σ_a^micro · φ_i · N_i
  （σ_a^micro 以 barn 给出；t 以秒计，接口的时间步以天计）。
- νΣf 与 Σa 中易裂变那一部分按剩余份额 s = N/N0 同步缩小；
  Σa 的其余部分与 D 保持不变。
- 每个时刻通量都归一到工况（所绑定版本）给定的总裂变率——恒定功率运行，
  燃料越烧通量幅值越高。

空间分辨：通量沿空间变化 ⇒ 同一区里不同位置消耗快慢不同，"一区一套截面"
从第一步起不再成立。剩余份额 s 逐单元存储，截面逐单元缩放；离散/求解器
本来就工作在逐单元截面数组上（见 discretization.Mesh），无需改动数值内核。

时间积分格式（只实现这一种，取舍见 README"燃耗时间积分"）：
步首冻结通量 + 步内解析指数积分（恒定通量耗散方程的精确解）：

    s_i ← s_i · exp(−σ_a^micro · φ_i · Δt)

冻结通量下该 ODE 的解析解就是指数；格式无条件保持 s ∈ (0, 1]、
不会出现负的数密度，全局一阶精度：时间步减半，首项误差减半。
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .discretization import Mesh, Region

SECONDS_PER_DAY = 86_400.0
BARN_TO_CM2 = 1.0e-24


def has_burnup(regions: list[Region]) -> bool:
    """是否有任一区声明了燃耗参数。"""
    return any(r.burnup is not None for r in regions)


def initial_remaining(mesh: Mesh) -> np.ndarray:
    """全场剩余份额 s = N/N0 初始为 1（不燃耗的单元恒为 1）。"""
    return np.ones(mesh.n, dtype=float)


def deplete_step(mesh: Mesh, regions: list[Region], remaining: np.ndarray,
                 phi: np.ndarray, dt_days: float) -> np.ndarray:
    """按步首通量 φ 推进一个时间步（天），返回新的逐单元剩余份额。

    只在声明了燃耗参数的区里消耗；其余单元原样保留（视为不燃耗）。
    """
    if dt_days <= 0.0 or not math.isfinite(dt_days):
        raise ValueError(f"时间步长必须为正有限值（天），收到 {dt_days!r}")
    if remaining.shape != (mesh.n,) or phi.shape != (mesh.n,):
        raise ValueError("剩余份额/通量与网格长度不一致")
    new = np.array(remaining, dtype=float, copy=True)
    lo = 0
    for reg, hi in zip(regions, mesh.region_hi):
        b = reg.burnup
        if b is not None:
            expo = (b.micro_sigma_a * BARN_TO_CM2
                    * phi[lo:hi] * (dt_days * SECONDS_PER_DAY))
            new[lo:hi] = remaining[lo:hi] * np.exp(-expo)
        lo = hi
    return new


def depleted_mesh(mesh: Mesh, regions: list[Region],
                  remaining: np.ndarray) -> Mesh:
    """按逐单元剩余份额缩放截面，返回新网格。

    νΣf ← νΣf·s；Σa ← Σa·(1 − f + f·s)（f 为 Σa 的易裂变份额）；
    Σa 其余部分与 D 不变。未声明燃耗的区保持原值。
    """
    sa = np.array(mesh.sigma_a, dtype=float, copy=True)
    nsf = np.array(mesh.nu_sigma_f, dtype=float, copy=True)
    lo = 0
    for reg, hi in zip(regions, mesh.region_hi):
        b = reg.burnup
        if b is not None:
            s = remaining[lo:hi]
            f = b.fissile_absorption_fraction
            nsf[lo:hi] = reg.nu_sigma_f * s
            sa[lo:hi] = reg.sigma_a * ((1.0 - f) + f * s)
        lo = hi
    return replace(mesh, sigma_a=sa, nu_sigma_f=nsf)


def region_remaining(mesh: Mesh, regions: list[Region],
                     remaining: np.ndarray) -> list[float]:
    """各区剩余易裂变份额（按单元宽度加权的体积平均）；不燃耗的区恒为 1。"""
    out: list[float] = []
    lo = 0
    for reg, hi in zip(regions, mesh.region_hi):
        if reg.burnup is None:
            out.append(1.0)
        else:
            w = mesh.widths[lo:hi]
            out.append(float(np.dot(w, remaining[lo:hi]) / w.sum()))
        lo = hi
    return out
