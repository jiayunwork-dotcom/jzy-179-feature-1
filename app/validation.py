"""物理规则校验：把跨字段错误聚合成逐项（字段 -> 原因）的清单。

拒收清单（需求逐条对应）：
- 厚度或 D ≤ 0            → 字段级 gt=0（models 已兜 422），这里对补丁/版本再验
- Σa、νΣf 任一为负         → ge=0 字段级 + 这里聚合
- 所有区都不含裂变材料     → 跨区：至少一区 νΣf > 0
- 区数多于 50
- 各区网格加起来多于 20000
"""
from __future__ import annotations

from typing import Any

MAX_REGIONS = 50
MAX_TOTAL_MESH = 20_000


def _regions_from_payload(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        regions = payload.get("regions", [])
    else:
        regions = getattr(payload, "regions", None)
        if regions is not None:
            regions = [r.model_dump() if hasattr(r, "model_dump") else vars(r)
                       for r in regions]
    return regions or []


def validate_regions(regions: list[dict]) -> list[tuple[str, str]]:
    """返回 [(field, message), ...]，空列表表示通过。"""
    errors: list[tuple[str, str]] = []

    if len(regions) == 0:
        errors.append(("regions", "至少需要一个区"))
        return errors
    if len(regions) > MAX_REGIONS:
        errors.append(("regions",
                       f"区数 {len(regions)} 超过上限 {MAX_REGIONS}"))

    total_mesh = 0
    any_fissile = False
    for i, r in enumerate(regions):
        prefix = f"regions[{i}]"
        thickness = r.get("thickness")
        D = r.get("D")
        sa = r.get("sigma_a")
        nsf = r.get("nu_sigma_f")
        n_mesh = r.get("n_mesh")

        if thickness is None or thickness <= 0:
            errors.append((f"{prefix}.thickness",
                           f"厚度必须 > 0，收到 {thickness!r}"))
        if D is None or D <= 0:
            errors.append((f"{prefix}.D",
                           f"扩散系数 D 必须 > 0，收到 {D!r}"))
        if sa is None or sa < 0:
            errors.append((f"{prefix}.sigma_a",
                           f"Σa 不允许为负，收到 {sa!r}"))
        if nsf is None or nsf < 0:
            errors.append((f"{prefix}.nu_sigma_f",
                           f"νΣf 不允许为负，收到 {nsf!r}"))
        if n_mesh is None or not isinstance(n_mesh, int) or n_mesh < 1:
            errors.append((f"{prefix}.n_mesh",
                           f"网格数必须为 ≥1 的整数，收到 {n_mesh!r}"))
        else:
            total_mesh += n_mesh
        if isinstance(nsf, (int, float)) and nsf > 0:
            any_fissile = True

    if total_mesh > MAX_TOTAL_MESH:
        errors.append(("regions",
                       f"各区网格总数 {total_mesh} 超过上限 {MAX_TOTAL_MESH}"))
    if not any_fissile:
        errors.append(("regions",
                       "所有区都不含裂变材料：至少需要一个区 νΣf > 0"))
    return errors


def validate_payload(payload: Any) -> list[tuple[str, str]]:
    return validate_regions(_regions_from_payload(payload))
