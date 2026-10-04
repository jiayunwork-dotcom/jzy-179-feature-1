"""均匀裸板解析参考解。

无限介质倍增因子与扩散长度平方：

    k∞ = νΣf / Σa,   L² = D / Σa

扩散方程 `-D φ'' + Σa φ = (νΣf/k) φ` 在单区均匀平板上，基模为余弦/正弦，
临界关系统一写成

    k = k∞ / (1 + L² B²)

几何曲率 B 由两端边界条件决定。
- 零通量（φ 在物理面上为零）：B = π / a（两端），B = π/(2a)（一端反射）。
- 外推边界（面外 d=2.13D 处 φ=0 的扩散理论/Marshak 处理）：有限体积在
  边界面上给出的连续极限是 Robin 条件 `J = D φ / d`，对应超越方程
      两端外推：   B·tan(Ba/2) = 1/d
      外推+反射：  B·tan(B a ) = 1/d
  本模块给出它的精确根（手写二分）。教科书常用的 B = π/(a+2d)、
  π/[2(a+d)] 是小曲率近似（Ba≪1 时两者一致到 O((Ba)^3)），一并给出。
- 两端反射且有裂变：B=0，k = k∞。

两端全反射单区均匀板的燃耗解析参考见 uniform_burnup_reference：
恒定总裂变率下 φ·N 为常数，数密度随时间线性下降，k(t) 随之有理式变化。
"""
from __future__ import annotations

import math

from .depletion import BARN_TO_CM2, SECONDS_PER_DAY
from .discretization import (
    EXTRAPOLATED,
    REFLECTIVE,
    ZERO_FLUX,
    Region,
)

EXTRAPOLATION_D = 2.13


def k_infinity(reg: Region) -> float:
    if reg.sigma_a <= 0.0:
        return math.inf if reg.nu_sigma_f > 0 else 0.0
    return reg.nu_sigma_f / reg.sigma_a


def l_squared(reg: Region) -> float:
    if reg.sigma_a <= 0.0:
        return math.inf
    return reg.D / reg.sigma_a


def _bisect_root(func, lo: float, hi: float, tol: float = 1e-13,
                 max_iter: int = 200) -> float:
    flo, fhi = func(lo), func(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("二分区间两端不变号")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if fm == 0.0 or 0.5 * (hi - lo) < tol:
            return mid
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _robin_buckling(thickness: float, d: float, *, reflected_end: bool) -> float:
    """Robin 真空边界的基模 B（手写二分求超越方程的根）。

    reflected_end=False：两端真空，B tan(Ba/2) = 1/d
    reflected_end=True ：一端真空一端反射，B tan(B a) = 1/d
    根位于 (0, π/a)（两端）或 (0, π/(2a))（单端）内。
    """
    if reflected_end:
        hi = math.pi / (2.0 * thickness)
        return _bisect_root(lambda B: B * math.tan(B * thickness) - 1.0 / d,
                            1e-12, hi * (1 - 1e-10))
    hi = math.pi / thickness
    return _bisect_root(lambda B: B * math.tan(0.5 * B * thickness) - 1.0 / d,
                        1e-12, hi * (1 - 1e-10))


def geometric_buckling(thickness: float, D: float, left: str, right: str,
                       *, approximate: bool = False) -> float:
    """基模 B。

    approximate=False（默认）：与有限体积外推边界一致的 Robin 精确根。
    approximate=True：教科书外推近似 π/(a+2d) 形式。
    """
    vacuum = (ZERO_FLUX, EXTRAPOLATED)
    d = EXTRAPOLATION_D * D

    if left == REFLECTIVE and right == REFLECTIVE:
        return 0.0

    n_ext = (1 if left == EXTRAPOLATED else 0) + (1 if right == EXTRAPOLATED else 0)
    n_refl = (1 if left == REFLECTIVE else 0) + (1 if right == REFLECTIVE else 0)

    if n_refl == 0 and left in vacuum and right in vacuum:
        if n_ext == 0:
            return math.pi / thickness                      # zero/zero
        if approximate:
            return math.pi / (thickness + n_ext * d)
        return _robin_buckling(thickness, d, reflected_end=False)

    if (left in vacuum or right in vacuum) and n_refl == 1:
        if n_ext == 0:
            return math.pi / (2.0 * thickness)              # zero/refl
        if approximate:
            return math.pi / (2.0 * (thickness + d))
        return _robin_buckling(thickness, d, reflected_end=True)

    raise ValueError(f"不支持的边界组合: {left}, {right}")


def analytical_k(reg: Region, left: str, right: str,
                 *, approximate: bool = False) -> float:
    """k∞ / (1 + L² B²)；无吸收体系按极限处理。"""
    kinf = k_infinity(reg)
    b2 = geometric_buckling(reg.thickness, reg.D, left, right,
                            approximate=approximate) ** 2
    if b2 == 0.0:
        return kinf
    l2 = l_squared(reg)
    if math.isinf(l2):
        return 0.0
    return kinf / (1.0 + l2 * b2)


def uniform_burnup_reference(*, nu_sigma_f: float, sigma_a: float,
                             fissile_absorption_fraction: float,
                             micro_sigma_a: float,
                             target_total_fission: float,
                             thickness: float,
                             time_days: float) -> tuple[float, float]:
    """两端全反射单区均匀板（B=0，无空间效应）的燃耗手推参考解。

    恒定总裂变率 F 下，通量被归一到 νΣf(t)·φ(t)·a = F，即
    φ·s = F/(νΣf0·a) 为常数（s = N/N0）。于是消耗方程

        ds/dt = −σ_a^micro·φ·s = −σ_a^micro·F/(νΣf0·a) = 常数

    给出**线性下降**的剩余份额（t 以秒计，接口时间以天计）：

        s(t) = 1 − σ_a^micro·F·t / (νΣf0·a)

    B=0 时 k = νΣf/Σa，截面随 s 缩放后：

        k(t) = νΣf0·s / (Σa0·(1−f) + Σa0·f·s)，f 为 Σa 的易裂变份额

    返回 (s, k)。s 降到 0 以下说明燃料已烧穿（截断到 0）。
    """
    rate = (micro_sigma_a * BARN_TO_CM2 * target_total_fission
            / (nu_sigma_f * thickness))          # 1/s
    s = 1.0 - rate * (time_days * SECONDS_PER_DAY)
    s = max(s, 0.0)
    f = fissile_absorption_fraction
    sa = sigma_a * ((1.0 - f) + f * s)
    k = (nu_sigma_f * s / sa) if sa > 0.0 else 0.0
    return s, k
