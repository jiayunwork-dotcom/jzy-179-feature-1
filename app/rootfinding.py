"""手写二分法求根（临界搜索内核）。

f(x) 单调（待定量为厚度或 νΣf 时，k 随其单调变化），调用方保证两端异号；
本模块只负责机械地二分、回报进度、响应取消。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Bracket:
    lo: float
    hi: float
    f_lo: float
    f_hi: float

    def width(self) -> float:
        return self.hi - self.lo


@dataclass
class RootResult:
    converged: bool
    root: float
    f_root: float
    iterations: int
    bracket: Bracket
    message: str
    cancelled: bool = False


def same_sign(a: float, b: float) -> bool:
    return (a > 0.0 and b > 0.0) or (a < 0.0 and b < 0.0)


def bisection(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    f_lo: float,
    f_hi: float,
    *,
    ftol: float = 1e-6,
    xtol: float = 1e-10,
    max_iter: int = 60,
    on_progress: Optional[Callable[[int, Bracket, float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> RootResult:
    """二分求 f(x)=0。

    参数
    ----
    func: 给 x 返回 f(x)=k(x)−1，要求对 [lo,hi] 单调。
    f_lo/f_hi: 两端已算好的函数值（由作业层先做两端校核，避免重复求解）。
    ftol: |f(root)| < ftol 即成功（要求 1e-6）。
    xtol: 区间缩到比 xtol 还窄时作为数值下限保护。
    max_iter: 二分步数上限；用完仍不满足 ftol 则 unconverged，
              回报最后区间与残差。
    on_progress(step, bracket, f_mid): 每步回调（写作业进度）。
    is_cancelled: 返回 True 时立即干净停下，cancelled=True，不再写结果。
    """
    if lo >= hi:
        raise ValueError("搜索区间必须满足 lo < hi")
    if same_sign(f_lo, f_hi) or f_lo == 0.0 or f_hi == 0.0:
        # 0 是恰好命中，但作业层把端点命中单独当成功处理
        if f_lo == 0.0:
            return RootResult(True, lo, 0.0, 0, Bracket(lo, hi, f_lo, f_hi),
                              "下端点恰为根")
        if f_hi == 0.0:
            return RootResult(True, hi, 0.0, 0, Bracket(lo, hi, f_lo, f_hi),
                              "上端点恰为根")
        raise ValueError("两端函数值同号，不能开始二分")

    bracket = Bracket(lo, hi, f_lo, f_hi)

    for step in range(1, max_iter + 1):
        if is_cancelled is not None and is_cancelled():
            return RootResult(
                False, 0.5 * (bracket.lo + bracket.hi), 0.0,
                step - 1, bracket, "作业已取消", cancelled=True,
            )

        mid = 0.5 * (bracket.lo + bracket.hi)
        f_mid = func(mid)

        if abs(f_mid) < ftol:
            if on_progress is not None:
                on_progress(step, bracket, f_mid)
            return RootResult(True, mid, f_mid, step, bracket,
                              f"第 {step} 步满足 |k−1| < {ftol:g}")

        if same_sign(f_mid, bracket.f_lo):
            bracket = Bracket(mid, bracket.hi, f_mid, bracket.f_hi)
        else:
            bracket = Bracket(bracket.lo, mid, bracket.f_lo, f_mid)

        if on_progress is not None:
            on_progress(step, bracket, f_mid)

        if bracket.width() <= xtol:
            # 区间已到数值精度但残差还不达标：如实报未收敛
            root = 0.5 * (bracket.lo + bracket.hi)
            return RootResult(False, root, min(abs(bracket.f_lo), abs(bracket.f_hi)),
                              step, bracket,
                              f"区间宽度 {bracket.width():.3e} 已小于 xtol，"
                              f"但残差未达 {ftol:g}")

    root = 0.5 * (bracket.lo + bracket.hi)
    return RootResult(
        False, root, min(abs(bracket.f_lo), abs(bracket.f_hi)), max_iter,
        bracket,
        f"二分 {max_iter} 步后仍未满足 |k−1| < {ftol:g}",
    )
