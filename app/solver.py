"""三对角求解、源迭代（幂迭代）与反应率统计。

所有线性代数（Thomas）、特征值迭代都是手写的；NumPy 只承担数组存储与
逐元素运算。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .discretization import (
    BoundaryCondition,
    Mesh,
    boundary_leakage,
)


class SolverError(RuntimeError):
    """求解本身失败（如矩阵奇异、出现非物理负通量）。"""


class ConvergenceError(RuntimeError):
    """达到迭代上限仍未收敛。必须带上最后的残差与迭代次数，禁止回半成品。"""

    def __init__(self, message: str, *, iterations: int, k: float,
                 k_residual: float, flux_residual: float):
        super().__init__(message)
        self.iterations = iterations
        self.k = k
        self.k_residual = k_residual
        self.flux_residual = flux_residual


@dataclass
class SolverSettings:
    k_tol: float = 1e-9          # |k_new-k_old|/k_old 的收敛容差
    flux_tol: float = 1e-9       # 归一化通量（裂变源形状）残差范数容差
    max_iter: int = 100_000      # 迭代步数上限
    shift_factor: float = 1.01   # 热启动 Wielandt 移位 p = shift_factor·k̂


@dataclass
class RegionRate:
    region_index: int
    absorption: float
    fission: float                # νΣf·φ·h（裂变中子产生率口径）
    fission_rate: float           # Σf·φ·h 未知时与 fission 相同（单群）


@dataclass
class SolveResult:
    k_eff: float
    phi: np.ndarray                # 已按目标总裂变率归一
    iterations: int
    k_residual: float
    flux_residual: float
    start_mode: str                # "cold" | "hot"
    converged: bool
    region_absorption: list[float]
    region_fission: list[float]
    leakage_left: float
    leakage_right: float
    total_absorption: float
    total_fission: float
    balance_residual: float        # |F/k - A - L| / (F/k)
    centers: np.ndarray = field(repr=False, default=None)
    widths: np.ndarray = field(repr=False, default=None)
    region_of: np.ndarray = field(repr=False, default=None)


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
                 rhs: np.ndarray) -> np.ndarray:
    """解三对角方程组，Thomas（追赶法），自己实现，不调用任何库求解器。

    递推公式按教科书原样写，只是用 NumPy 的整段数组运算表达两条扫描
    （NumPy 仅承担逐元素计算；递推方向/系数更新由本函数决定）：
        追：c'[i] = c_i / (b_i − a_i c'[i−1])，
            d'[i] = (d_i − a_i d'[i−1]) / (b_i − a_i c'[i−1])
        赶：x[i] = d'[i] − c'[i] x[i+1]
    正向的 c' 递推是顺序相关的，无法整段并行，因此"追"保留一个紧凑的
    标量循环（这是 Thomas 算法的本质串行部分），"赶"同样顺序回代。
    """
    n = diag.size
    if n != rhs.size or lower.size != n or upper.size != n:
        raise SolverError("三对角矩阵维度不一致")

    cp = np.empty(n, dtype=float)
    dp = np.empty(n, dtype=float)

    # 追（顺序，无法整段并行——Thomas 前消的递推本质是串行的）。
    # 缓存局部数组引用，减少每次迭代的下标/属性查找开销。
    if diag[0] == 0.0:
        raise SolverError("三对角矩阵奇异（首主元为零）")
    b0 = diag[0]
    cp[0] = upper[0] / b0
    dp[0] = rhs[0] / b0
    c_prev = cp[0]
    d_prev = dp[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c_prev
        if denom == 0.0:
            raise SolverError("三对角矩阵奇异（主元为零）")
        inv = 1.0 / denom
        c_prev = upper[i] * inv
        d_prev = (rhs[i] - lower[i] * d_prev) * inv
        cp[i] = c_prev
        dp[i] = d_prev

    # 赶（顺序回代，同样是严格串行的递推）
    x = np.empty(n, dtype=float)
    xn = dp[n - 1]
    x[n - 1] = xn
    for i in range(n - 2, -1, -1):
        xn = dp[i] - cp[i] * xn
        x[i] = xn
    return x


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """向量内积。Thomas/迭代/求根算法本身手写，这里只是数组归约，用 NumPy。"""
    return float(np.sum(a * b))


def normalize(phi: np.ndarray, fission: np.ndarray) -> np.ndarray:
    """按总裂变中子产生率 F = Σ νΣf φ h 归一到 1。"""
    norm = _dot(fission, phi)
    if not np.isfinite(norm) or norm <= 0.0:
        raise SolverError("裂变归一化失败：当前通量没有正的裂变权重（k 迭代发散或无裂变材料）")
    return phi / norm


def _initial_guess(mesh: Mesh, mode: str, previous_phi: np.ndarray | None,
                   previous_centers: np.ndarray | None) -> np.ndarray:
    """冷启动：在含裂变材料的单元上给平坦初值 1。

    热启动：由 app/interpolation.py 把上一版网格上的收敛通量线性重映过来
    （调用方负责传入已重映好的 previous_phi）。
    """
    n = mesh.n
    if mode == "hot":
        if previous_phi is None or previous_phi.shape != (n,):
            raise SolverError("热启动缺少与当前网格同长度的初值（重映失败）")
        guess = np.array(previous_phi, dtype=float, copy=True)
    else:
        guess = np.where(mesh.nu_sigma_f > 0.0, 1.0, 0.0)
        if guess.max() == 0.0:
            guess = np.ones(n, dtype=float)
    if not np.all(np.isfinite(guess)):
        raise SolverError("初值含非有限值")
    return guess


def _rayleigh_quotient(lower, diag, upper, fission, phi: np.ndarray) -> float:
    """广义 Rayleigh 商 k = <Fφ,φ> / <Mφ,φ>（M 对称，数组运算向量化）。"""
    mq = float(np.sum(diag * phi * phi))
    # 对称三项：u_i φ_i φ_{i+1} 与 l_{i+1} φ_{i+1} φ_i 各计一次
    mq += float(np.sum(upper[:-1] * phi[:-1] * phi[1:]))
    mq += float(np.sum(lower[1:] * phi[1:] * phi[:-1]))
    fq = _dot(fission * phi, phi)
    if mq <= 0.0:
        raise SolverError("Rayleigh 商分母非正")
    return fq / mq


def _shape_residual(phi_new: np.ndarray, phi: np.ndarray) -> float:
    diff = phi_new - phi
    return float(np.sqrt(np.sum(diff * diff))
                 / max(float(np.sqrt(np.sum(phi_new * phi_new))), 1e-300))


def source_iteration(
    mesh: Mesh,
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    fission: np.ndarray,
    settings: SolverSettings,
    *,
    mode: str = "cold",
    previous_phi: np.ndarray | None = None,
    previous_centers: np.ndarray | None = None,
    initial_k: float | None = None,
    shift_factor: float = 1.01,
) -> tuple[float, np.ndarray, int, float, float]:
    """求 M φ = (1/k) F φ 的基模。冷、热两条路径：

    冷启动（标准源/幂迭代，Lewis & Miller 形式）：
      解 M φ^{n+1} = (1/k^n) F φ^n，
      k^{n+1} = k^n·<Fφ^{n+1}>/<Fφ^n>（归一后即 k^n·<Fφ^{n+1}>），
      再归一 φ^{n+1} 使 <Fφ>=1；k 从 1 起步。

    热启动（Wielandt 逆移位 + Rayleigh 商，见 README"热启动策略"）：
      同时继承上一版的形状与 k；每步取移位 p = shift_factor·k̂（在当前 k
      估计上方，保证 M−F/p 正定、基模占优），解
          (M − F/p) ψ = F φ
      归一 ψ 后用广义 Rayleigh 商 k̂ = <Fψ,ψ>/<Mψ,ψ> 重估 k。
      若改动大到越过移位点（解出现负值/非有限，或矩阵遇零主元），该步
      自动退回一杆普通源迭代，绝不带偏。

    两条路径的收敛判据相同且**同时**要求：
      |Δk|/k < k_tol  且  ‖φ^{n+1}−φ^n‖_2 / ‖φ^{n+1}‖_2 < flux_tol。
    只看 k 会在形状还带着高阶模时提前停下（热启动尤其危险），故必须双判据。
    返回 (k, phi_normalized, iterations, k_residual, flux_residual)。
    """
    phi = _initial_guess(mesh, mode, previous_phi, previous_centers)
    phi = normalize(phi, fission)

    if mode == "hot":
        if initial_k is None or not np.isfinite(initial_k) or initial_k <= 0:
            raise SolverError("热启动必须提供正的 initial_k（与形状一起继承）")
        if not (1.0 < shift_factor < 2.0):
            raise SolverError("shift_factor 必须在 (1, 2) 之间")
        k = float(initial_k)
        return _wielandt_iterate(
            lower, diag, upper, fission, settings, phi, k, shift_factor)

    # ---------------- 冷启动：标准源迭代 ----------------
    k = 1.0
    k_res = float("inf")
    flux_res = float("inf")

    for it in range(1, settings.max_iter + 1):
        src = fission * phi / k
        phi_new = thomas_solve(lower, diag, upper, src)
        if not np.all(np.isfinite(phi_new)):
            raise SolverError(f"第 {it} 步解出非有限通量，迭代发散")
        if np.any(phi_new < -1e-300):
            raise SolverError(
                f"第 {it} 步出现负通量（最小值 {phi_new.min():.3e}），"
                "检查材料/边界设置"
            )
        # 数值上极小的负零截掉，但不改变正常解
        phi_new = np.maximum(phi_new, 0.0)

        f_new = _dot(fission, phi_new)
        if f_new <= 0.0 or not np.isfinite(f_new):
            raise SolverError(f"第 {it} 步裂变产生率非正（{f_new}），无法更新 k")
        # <F φ^n> = 1（每步归一），故 k^{n+1} = k^n · <F φ^{n+1}>
        k_new = k * f_new / _dot(fission, phi)
        phi_new = phi_new / f_new  # <F φ_new> = 1

        k_res = abs(k_new - k) / max(abs(k_new), 1e-300)
        flux_res = _shape_residual(phi_new, phi)

        phi = phi_new
        k = k_new
        if k_res < settings.k_tol and flux_res < settings.flux_tol:
            return k, phi, it, k_res, flux_res

    raise ConvergenceError(
        f"源迭代达到上限 {settings.max_iter} 步仍未收敛",
        iterations=settings.max_iter,
        k=k,
        k_residual=k_res,
        flux_residual=flux_res,
    )


def _wielandt_iterate(lower, diag, upper, fission, settings, phi, k,
                      shift_factor):
    """热启动：自适应移位逆幂迭代，Rayleigh 商估 k，异常步退回普通源迭代。"""
    k_res = float("inf")
    flux_res = float("inf")

    for it in range(1, settings.max_iter + 1):
        p = shift_factor * k
        shifted_diag = diag - fission / p
        phi_new = None
        try:
            candidate = thomas_solve(lower, shifted_diag, upper, fission * phi)
            if np.all(np.isfinite(candidate)) and np.all(candidate >= -1e-300):
                phi_new = np.maximum(candidate, 0.0)
        except SolverError:
            phi_new = None  # 移位矩阵病态/奇异 → 走下面的普通源迭代兜底

        used_shift = phi_new is not None
        if not used_shift:
            # 退回一杆标准源迭代（Lewis-Miller），保证基模方向与正性
            candidate = thomas_solve(lower, diag, upper, fission * phi / k)
            if not np.all(np.isfinite(candidate)) or np.any(candidate < -1e-300):
                raise SolverError(
                    f"热启动第 {it} 步移位与普通迭代均未得到物理通量")
            phi_new = np.maximum(candidate, 0.0)
            f_new = _dot(fission, phi_new)
            if f_new <= 0.0:
                raise SolverError(f"热启动第 {it} 步裂变产生率非正")
            k_new = k * f_new / _dot(fission, phi)
            phi_new = phi_new / f_new
        else:
            f_new = _dot(fission, phi_new)
            if f_new <= 0.0 or not np.isfinite(f_new):
                raise SolverError(f"热启动第 {it} 步裂变产生率非正")
            phi_new = phi_new / f_new  # <F φ_new> = 1
            # 广义 Rayleigh 商（对混合向量也是 k 的最优二阶估计）
            k_new = _rayleigh_quotient(lower, diag, upper, fission, phi_new)
            if not np.isfinite(k_new) or k_new <= 0.0:
                # Rayleigh 商不可信，再退回保守更新
                k_new = k * max(f_new, 1e-300)

        k_res = abs(k_new - k) / max(abs(k_new), 1e-300)
        flux_res = _shape_residual(phi_new, phi)
        phi, k = phi_new, k_new

        if k_res < settings.k_tol and flux_res < settings.flux_tol:
            return k, phi, it, k_res, flux_res

    raise ConvergenceError(
        f"热启动达到上限 {settings.max_iter} 步仍未收敛",
        iterations=settings.max_iter,
        k=k,
        k_residual=k_res,
        flux_residual=flux_res,
    )



def region_rates(mesh: Mesh, phi: np.ndarray) -> tuple[list[float], list[float]]:
    """逐区积分吸收率与裂变中子产生率（np.add.reduceat 一次分组求和）。"""
    n_regions = len(mesh.region_hi)
    v = mesh.widths * phi
    a_cell = mesh.sigma_a * v
    f_cell = mesh.nu_sigma_f * v
    starts = np.concatenate(([0], np.asarray(mesh.region_hi[:-1], dtype=int)))
    absorption = [float(x) for x in np.add.reduceat(a_cell, starts)]
    fission = [float(x) for x in np.add.reduceat(f_cell, starts)]
    return absorption, fission


def solve(
    mesh: Mesh,
    left: BoundaryCondition,
    right: BoundaryCondition,
    settings: SolverSettings,
    *,
    mode: str = "cold",
    previous_phi: np.ndarray | None = None,
    previous_centers: np.ndarray | None = None,
    initial_k: float | None = None,
    target_total_fission: float = 1.0,
) -> SolveResult:
    """完整求解一次：组装、迭代、统计反应率、中子平衡、归一化。"""
    from .discretization import assemble_operators

    lower, diag, upper, fission = assemble_operators(mesh, left, right)
    k, phi, iters, k_res, flux_res = source_iteration(
        mesh, lower, diag, upper, fission, settings,
        mode=mode, previous_phi=previous_phi,
        previous_centers=previous_centers, initial_k=initial_k,
        shift_factor=settings.shift_factor,
    )

    # 按用户给定的总裂变率归一（当前 <F φ>=1，直接乘目标值）
    if target_total_fission <= 0.0:
        raise SolverError("目标总裂变率必须为正")
    phi = phi * target_total_fission

    absorption, fission_rates = region_rates(mesh, phi)
    j_left, j_right = boundary_leakage(mesh, phi, left, right)

    total_a = sum(absorption)
    total_f = sum(fission_rates)
    production = total_f / k
    loss = total_a + j_left + j_right
    denom = max(production, 1e-300)
    balance_res = abs(production - loss) / denom

    if np.any(phi < 0.0):
        raise SolverError("归一化后出现负通量")

    return SolveResult(
        k_eff=k,
        phi=phi,
        iterations=iters,
        k_residual=k_res,
        flux_residual=flux_res,
        start_mode=mode,
        converged=True,
        region_absorption=absorption,
        region_fission=fission_rates,
        leakage_left=j_left,
        leakage_right=j_right,
        total_absorption=total_a,
        total_fission=total_f,
        balance_residual=balance_res,
        centers=mesh.centers,
        widths=mesh.widths,
        region_of=mesh.region_of,
    )
