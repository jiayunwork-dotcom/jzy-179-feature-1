"""有限体积离散与边界条件。

方程（1D slab，单群，无外源）：

    -d/dx (D dφ/dx) + Σa φ = (1/k) νΣf φ

采用单元中心有限体积（守恒型）离散。单元 i 宽 h_i、中心存 φ_i。
内界面 i+1/2 上只有"一个"净流，左右两个单元的平衡方程中以相反符号出现
同一项，因此界面上通量连续、净流 -D dφ/dx 连续是构造性满足的——
两侧 D、Σa 从不做算术平均。界面传导率取调和平均（两点格式）：

    J_{i+1/2} = g_{i+1/2} (φ_i − φ_{i+1})
    g_{i+1/2} = 2 D_i D_{i+1} / (D_i h_{i+1} + D_{i+1} h_i)

同区等距时 g = D/h，退化为标准中心差分，裸板误差 O(h^2)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 边界类型
ZERO_FLUX = "zero_flux"          # 物理面上通量直接为零
EXTRAPOLATED = "extrapolated"    # 外推距离 2.13 D 处通量为零
REFLECTIVE = "reflective"        # 全反射（对称面），净流为零

BOUNDARY_TYPES = (ZERO_FLUX, EXTRAPOLATED, REFLECTIVE)
EXTRAPOLATION_FACTOR = 2.13


@dataclass(frozen=True)
class Region:
    """一个材料区：厚度、D、Σa、νΣf、网格数。"""
    thickness: float
    D: float
    sigma_a: float
    nu_sigma_f: float
    n_mesh: int

    @property
    def h(self) -> float:
        return self.thickness / self.n_mesh


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in BOUNDARY_TYPES:
            raise ValueError(f"未知边界类型: {self.kind!r}，可选 {BOUNDARY_TYPES}")


@dataclass(frozen=True)
class Mesh:
    """全场网格信息。

    widths      各单元宽度 h_i
    centers     各单元中心坐标 x_i
    D/sa/nsf    逐单元材料参数
    region_of   单元 -> 区号
    region_hi   各区末单元下标（不含）
    """
    widths: np.ndarray
    centers: np.ndarray
    D: np.ndarray
    sigma_a: np.ndarray
    nu_sigma_f: np.ndarray
    region_of: np.ndarray
    region_hi: tuple[int, ...]

    @property
    def n(self) -> int:
        return self.widths.size

    @property
    def total_width(self) -> float:
        return float(self.widths.sum())


def build_mesh(regions: list[Region]) -> Mesh:
    """沿 x 依次拼接各区；区内均匀分网，界面两侧单元一一对应。"""
    widths_parts: list[np.ndarray] = []
    centers_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    sa_parts: list[np.ndarray] = []
    nsf_parts: list[np.ndarray] = []
    region_of_parts: list[np.ndarray] = []
    region_hi: list[int] = []

    x_edge = 0.0
    for r_idx, reg in enumerate(regions):
        h = reg.h
        widths_parts.append(np.full(reg.n_mesh, h, dtype=float))
        centers_parts.append(x_edge + (np.arange(reg.n_mesh) + 0.5) * h)
        d_parts.append(np.full(reg.n_mesh, reg.D, dtype=float))
        sa_parts.append(np.full(reg.n_mesh, reg.sigma_a, dtype=float))
        nsf_parts.append(np.full(reg.n_mesh, reg.nu_sigma_f, dtype=float))
        region_of_parts.append(np.full(reg.n_mesh, r_idx, dtype=int))
        x_edge += reg.thickness
        region_hi.append(sum(r.n_mesh for r in regions[: r_idx + 1]))

    return Mesh(
        widths=np.concatenate(widths_parts),
        centers=np.concatenate(centers_parts),
        D=np.concatenate(d_parts),
        sigma_a=np.concatenate(sa_parts),
        nu_sigma_f=np.concatenate(nsf_parts),
        region_of=np.concatenate(region_of_parts),
        region_hi=tuple(region_hi),
    )


def inner_conductance(D_l: float, h_l: float, D_r: float, h_r: float) -> float:
    """内界面调和平均传导率（界面两侧截面不平均）。"""
    return 2.0 * D_l * D_r / (D_l * h_r + D_r * h_l)


def boundary_conductance(kind: str, D: float, h: float) -> float:
    """外边界面（左/右端最外侧单元的半宽 h/2）的传导率。"""
    if kind == REFLECTIVE:
        return 0.0
    if kind == ZERO_FLUX:
        # 镜像零值点恰在物理面上，面到中心距离 h/2 → g = 2D/h
        return 2.0 * D / h
    if kind == EXTRAPOLATED:
        # 零值点在物理面外 2.13 D 处，距中心 h/2 + 2.13 D
        return D / (0.5 * h + EXTRAPOLATION_FACTOR * D)
    raise ValueError(f"未知边界类型: {kind!r}")


def assemble_operators(mesh: Mesh, left: BoundaryCondition, right: BoundaryCondition):
    """组装三对角损失矩阵 M 和对角裂变产生矩阵 F。

    单元平衡（M φ = (1/k) F φ）：

        g_{i-1/2}(φ_i−φ_{i-1}) + g_{i+1/2}(φ_i−φ_{i+1})
        + Σa_i h_i φ_i = (1/k) νΣf_i h_i φ_i

    外边界流出项以 g_b φ_i 计入对角（反射时 g_b=0）。

    返回 (lower, diag, upper, fission)：M 的下次对角/对角/上次对角
    及 F 的对角元，长度均为 n（lower[0]=upper[-1]=0）。
    """
    n = mesh.n
    lower = np.zeros(n, dtype=float)
    diag = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)
    fission = mesh.nu_sigma_f * mesh.widths

    # 内界面
    for i in range(n - 1):
        g = inner_conductance(
            mesh.D[i], mesh.widths[i], mesh.D[i + 1], mesh.widths[i + 1]
        )
        upper[i] = -g
        lower[i + 1] = -g
        diag[i] += g
        diag[i + 1] += g

    # 外边界（流出，非负进入对角）
    diag[0] += boundary_conductance(left.kind, mesh.D[0], mesh.widths[0])
    diag[n - 1] += boundary_conductance(
        right.kind, mesh.D[n - 1], mesh.widths[n - 1]
    )

    # 吸收
    diag += mesh.sigma_a * mesh.widths
    return lower, diag, upper, fission


def boundary_leakage(
    mesh: Mesh, phi: np.ndarray, left: BoundaryCondition, right: BoundaryCondition
) -> tuple[float, float]:
    """左右两端向外的净泄漏率 J·面积（单位面积，1D 即 J），均以流出为正。"""
    gl = boundary_conductance(left.kind, mesh.D[0], mesh.widths[0])
    gr = boundary_conductance(right.kind, mesh.D[-1], mesh.widths[-1])
    return float(gl * phi[0]), float(gr * phi[-1])
