"""热启动初值重映：把旧网格上的收敛通量映射到新网格。

手写分段线性插值（不调用 np.interp），落在旧坐标范围外的单元用最近的
边界值外延（反射层一类新增材料区里初值为正即可，第一步源迭代会立刻修正）。
"""
from __future__ import annotations

import numpy as np


def piecewise_linear_remap(
    old_centers: np.ndarray,
    old_phi: np.ndarray,
    new_centers: np.ndarray,
) -> np.ndarray:
    """分段线性重映。

    old_centers 必须单调递增。new_centers 落在区间外用端点值常数外延。
    结果保证非负（输入物理上非负，插值是凸组合；外延为正常数）。
    """
    if old_centers.shape != old_phi.shape or old_centers.size < 1:
        raise ValueError("重映输入形状不合法")
    for i in range(1, old_centers.size):
        if old_centers[i] <= old_centers[i - 1]:
            raise ValueError("旧网格中心坐标必须严格单调递增")

    out = np.empty(new_centers.size, dtype=float)
    n_old = old_centers.size
    j = 0
    for i, x in enumerate(new_centers):
        if x <= old_centers[0]:
            out[i] = old_phi[0]
            continue
        if x >= old_centers[-1]:
            out[i] = old_phi[-1]
            continue
        # 推进到 old_centers[j] <= x < old_centers[j+1]
        while j < n_old - 2 and x > old_centers[j + 1]:
            j += 1
        x0, x1 = old_centers[j], old_centers[j + 1]
        t = (x - x0) / (x1 - x0)
        val = (1.0 - t) * old_phi[j] + t * old_phi[j + 1]
        out[i] = max(val, 0.0)
    return out
