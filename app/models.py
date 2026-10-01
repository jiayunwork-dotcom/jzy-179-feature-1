"""Pydantic 接口模型。数值硬约束在这里给出 422 与字段定位；
跨字段/物理规则（含裂变、区数、网格总数）在 validation.py 聚合报错。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

BoundaryKind = Literal["zero_flux", "extrapolated", "reflective"]


class RegionIn(BaseModel):
    thickness: float = Field(..., gt=0, description="区厚度 (cm)，必须 > 0")
    D: float = Field(..., gt=0, description="扩散系数 (cm)，必须 > 0")
    sigma_a: float = Field(..., ge=0, description="宏观吸收截面 (cm^-1)，≥ 0")
    nu_sigma_f: float = Field(..., ge=0, description="νΣf (cm^-1)，≥ 0")
    n_mesh: int = Field(..., ge=1, description="该区网格数，≥ 1")


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
