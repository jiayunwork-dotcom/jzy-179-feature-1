"""Pydantic 接口模型。数值硬约束在这里给出 422 与字段定位；
跨字段/物理规则（含裂变、区数、网格总数）在 validation.py 聚合报错。
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

BoundaryKind = Literal["zero_flux", "extrapolated", "reflective"]


class RegionBurnup(BaseModel):
    """区的燃耗声明（可选）。未声明的区不燃耗，参数始终不变。"""
    initial_density: float = Field(
        ..., gt=0, description="易裂变核素初始数密度 (10^24/cm^3)，必须 > 0")
    micro_sigma_a: float = Field(
        ..., ge=0, description="微观吸收截面 (barn)，≥ 0；为 0 时不消耗")
    fissile_absorption_fraction: float = Field(
        ..., ge=0, le=1, description="Σa 中来自易裂变核素的份额，∈ [0,1]")


class RegionIn(BaseModel):
    thickness: float = Field(..., gt=0, description="区厚度 (cm)，必须 > 0")
    D: float = Field(..., gt=0, description="扩散系数 (cm)，必须 > 0")
    sigma_a: float = Field(..., ge=0, description="宏观吸收截面 (cm^-1)，≥ 0")
    nu_sigma_f: float = Field(..., ge=0, description="νΣf (cm^-1)，≥ 0")
    n_mesh: int = Field(..., ge=1, description="该区网格数，≥ 1")
    burnup: Optional[RegionBurnup] = None


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
    burnup: Optional[RegionBurnup] = None


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


# ---------- 燃耗历程 ----------

MAX_BURNUP_STEPS = 500
StepDays = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class BurnupCreate(BaseModel):
    """新建燃耗历程：挂在工况的某一版参数上，按给定时间步序列（天）推进。"""
    version: Optional[int] = Field(None, ge=1,
                                   description="绑定哪一版参数；缺省为当前版本")
    steps: list[StepDays] = Field(
        ..., min_length=1, max_length=MAX_BURNUP_STEPS,
        description="时间步序列（天），每步 > 0，至多 500 步")
    max_steps: Optional[int] = Field(
        None, ge=1, description="本次最多推进几步（分段推进用）；缺省推到底")
    note: Optional[str] = None


class BurnupAdvance(BaseModel):
    """继续推进一条未完成的历程。"""
    max_steps: Optional[int] = Field(
        None, ge=1, description="本次最多推进几步；缺省推到底")


class BurnupOut(BaseModel):
    id: str
    case_id: str
    case_name: str
    version: int
    status: str            # running|paused|cancelled|interrupted|completed|failed
    steps: list[float]
    total_steps: int
    steps_completed: int
    current_time_days: float
    current_k: Optional[float] = None
    cancelled: bool
    message: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    note: Optional[str] = None


class BurnupPointSummary(BaseModel):
    id: str
    step_index: int
    time_days: float
    k_eff: float


class BurnupPointOut(BaseModel):
    id: str
    burnup_id: str
    case_id: str
    version: int
    step_index: int
    time_days: float
    created_at: str
    k_eff: float
    iterations: int
    k_residual: float
    flux_residual: float
    region_remaining: list[float]
    cell_remaining: list[float]
    region_absorption: list[float]
    region_fission: list[float]
    leakage_left: float
    leakage_right: float
    total_absorption: float
    total_fission: float
    balance_residual: float
    target_total_fission: float
    n_mesh_total: int
    centers: list[float]
    widths: list[float]
    region_of: list[int]
    phi: list[float]


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
    burnup: Optional[RegionBurnup] = None


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
