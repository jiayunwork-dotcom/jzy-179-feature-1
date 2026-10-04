"""Pydantic 接口模型。数值硬约束在这里给出 422 与字段定位；
跨字段/物理规则（含裂变、区数、网格总数）在 validation.py 聚合报错。
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

BoundaryKind = Literal["zero_flux", "extrapolated", "reflective"]


class BurnupIn(BaseModel):
    """燃料区的燃耗声明（可选）。三个量必须一起给出。

    fissile_density_0            易裂变核素初始数密度 N0 (10^24 cm^-3)，> 0
    sigma_a_micro                易裂变核素微观吸收截面 (barn)，≥ 0；为 0 表示不消耗
    fissile_absorption_fraction  该区 Σa 中来自易裂变核素的份额，∈ [0, 1]
    """
    fissile_density_0: float = Field(..., gt=0, allow_inf_nan=False)
    sigma_a_micro: float = Field(..., ge=0, allow_inf_nan=False)
    fissile_absorption_fraction: float = Field(..., ge=0, le=1, allow_inf_nan=False)


class RegionIn(BaseModel):
    thickness: float = Field(..., gt=0, description="区厚度 (cm)，必须 > 0")
    D: float = Field(..., gt=0, description="扩散系数 (cm)，必须 > 0")
    sigma_a: float = Field(..., ge=0, description="宏观吸收截面 (cm^-1)，≥ 0")
    nu_sigma_f: float = Field(..., ge=0, description="νΣf (cm^-1)，≥ 0")
    n_mesh: int = Field(..., ge=1, description="该区网格数，≥ 1")
    burnup: Optional[BurnupIn] = Field(
        None, description="燃耗声明；缺省表示该区不燃耗")


class BoundaryIn(BaseModel):
    kind: BoundaryKind


class SolverSettingsIn(BaseModel):
    k_tol: Optional[float] = Field(None, gt=0, le=1e-2)
    flux_tol: Optional[float] = Field(None, gt=0, le=1e-2)
    max_iter: Optional[int] = Field(None, ge=1, le=10_000_000)


class CaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    regions: list[RegionIn] = Field(..., min_length=1)
    left_boundary: BoundaryKind = "zero_flux"
    right_boundary: BoundaryKind = "zero_flux"
    settings: SolverSettingsIn = Field(default_factory=SolverSettingsIn)
    target_total_fission: float = Field(1.0, gt=0)


class RegionPatch(BaseModel):
    """修改某一区；字段缺省即保持原值。"""
    thickness: Optional[float] = Field(None, gt=0)
    D: Optional[float] = Field(None, gt=0)
    sigma_a: Optional[float] = Field(None, ge=0)
    nu_sigma_f: Optional[float] = Field(None, ge=0)
    n_mesh: Optional[int] = Field(None, ge=1)
    burnup: Optional[BurnupIn] = None


class CasePatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    region_index: Optional[int] = Field(None, ge=0)
    region: Optional[RegionPatch] = None
    left_boundary: Optional[BoundaryKind] = None
    right_boundary: Optional[BoundaryKind] = None
    settings: Optional[SolverSettingsIn] = None
    target_total_fission: Optional[float] = Field(None, gt=0)


class SolveRequest(BaseModel):
    mode: Literal["cold", "hot"] = "cold"
    settings: Optional[SolverSettingsIn] = None
    target_total_fission: Optional[float] = Field(None, gt=0)
    note: Optional[str] = None


class SearchRequest(BaseModel):
    region_index: int = Field(..., ge=0)
    parameter: Literal["thickness", "nu_sigma_f"]
    lower: float
    upper: float = Field(...)
    max_steps: int = Field(50, ge=1, le=500)
    note: Optional[str] = None


# ---------- 燃耗 ----------

MAX_BURNUP_STEPS_PER_REQUEST = 500   # 单次提交/续推的步数上限（历程累计同样封顶 500）

# 时间步长（天）：正、有限；list 逐项校验
StepDays = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class BurnupCreate(BaseModel):
    """新建燃耗历程：挂在工况的某一版参数上，给出时间步序列（天）。"""
    steps: list[StepDays] = Field(
        ..., min_length=1, max_length=MAX_BURNUP_STEPS_PER_REQUEST)
    version: Optional[int] = Field(None, ge=1)
    note: Optional[str] = None


class BurnupAdvance(BaseModel):
    """在已有历程上接着往后推一段。"""
    steps: list[StepDays] = Field(
        ..., min_length=1, max_length=MAX_BURNUP_STEPS_PER_REQUEST)


# ---------- 响应模型 ----------

class SettingsOut(BaseModel):
    k_tol: float
    flux_tol: float
    max_iter: int


class RegionOut(BaseModel):
    thickness: float
    D: float
    sigma_a: float
    nu_sigma_f: float
    n_mesh: int
    burnup: Optional[BurnupIn] = None   # 老数据没有该键，读出为 null


class CaseSummary(BaseModel):
    id: str
    name: str
    current_version: int
    created_at: str
    updated_at: str


class CaseOut(BaseModel):
    id: str
    name: str
    current_version: int
    created_at: str
    updated_at: str
    regions: list[RegionOut]
    left_boundary: BoundaryKind
    right_boundary: BoundaryKind
    settings: SettingsOut
    target_total_fission: float
    result_ids: list[str]


class SolveFailure(BaseModel):
    ok: bool = False
    error: str
    iterations: int
    k: float
    k_residual: float
    flux_residual: float


class ResultOut(BaseModel):
    id: str
    case_id: str
    case_name: str
    version: int
    created_at: str
    mode: str
    k_eff: float
    iterations: int
    k_residual: float
    flux_residual: float
    target_total_fission: float
    region_absorption: list[float]
    region_fission: list[float]
    leakage_left: float
    leakage_right: float
    total_absorption: float
    total_fission: float
    balance_residual: float
    n_mesh_total: int
    centers: list[float]
    widths: list[float]
    region_of: list[int]
    phi: list[float]
    note: Optional[str] = None
    warm_from_result: Optional[str] = None


class JobOut(BaseModel):
    id: str
    case_id: str
    case_name: str
    version: int
    status: str                       # queued|running|converged|unconverged|rejected|cancelled|failed
    region_index: int
    parameter: str
    lower: float
    upper: float
    steps_taken: int
    max_steps: int
    current_bracket: Optional[tuple[float, float]] = None
    current_k: Optional[float] = None
    root_value: Optional[float] = None
    k_at_root: Optional[float] = None
    endpoint_k: Optional[tuple[float, float]] = None
    message: str
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancelled: bool
    result_id: Optional[str] = None
    note: Optional[str] = None


class ErrorItem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    error: str
    details: list[ErrorItem] = Field(default_factory=list)


# ---------- 燃耗响应 ----------

class BurnupOut(BaseModel):
    id: str
    case_id: str
    case_name: str
    version: int
    status: str                  # queued|running|completed|cancelled|interrupted|exhausted|failed
    steps_completed: int         # 已完成的时间步数（累计，跨分段）
    steps_planned: int           # 本段请求的步数
    n_points: int                # 已写入的时间点数（含 t=0 初始点）
    burnup_time_days: float      # 当前燃耗时间（最后一个已完成时间点的累计天数）
    current_k: Optional[float] = None   # 最后一个已完成时间点的 k
    cancelled: bool
    message: str
    note: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class BurnupPointSummary(BaseModel):
    id: str
    burnup_id: str
    step_index: int              # 0 为新装料初始点，i 为第 i 步之后
    time_days: float
    step_days: float
    k_eff: float
    region_remaining: list[float]
    balance_residual: float
    created_at: str


class BurnupPointOut(BaseModel):
    id: str
    burnup_id: str
    case_id: str
    case_name: str
    version: int
    step_index: int
    time_days: float
    step_days: float
    k_eff: float
    region_remaining: list[float]   # 各区剩余易裂变份额（体积加权）
    cell_remaining: list[float]     # 逐单元剩余份额（续推状态，同样公开）
    region_absorption: list[float]
    region_fission: list[float]
    leakage_left: float
    leakage_right: float
    total_absorption: float
    total_fission: float
    balance_residual: float
    n_mesh_total: int
    centers: list[float]
    widths: list[float]
    region_of: list[int]
    phi: list[float]
    created_at: str
