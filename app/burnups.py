"""燃耗历程调度：挂在工况某一版参数上，后台分段推进，可取消、可续算。

- 历程文档内嵌创建时刻所绑定版本的参数快照（regions 含燃耗声明、边界、
  求解设置、目标总裂变率）；工况随后被改、被删都不影响已创建的历程。
- 每个已完成时间点单独落盘（burnup_points/），历程文档只记进度与点索引。
  分段推进或服务重启后，从最后一个已落盘点的状态（逐单元剩余份额、
  归一化通量、k）原样继续——双精度浮点经 JSON 往返逐位不变，且求解
  过程完全确定，因此续算与一次推到底的结果逐点一致。
- 取消只影响未完成的步：已完成的点全部保留，cancelled/interrupted/paused
  三种状态都可用 advance 接着往后推。
- 单个时间步：步首冻结通量做解析指数耗散（depletion.deplete_step），
  再以上一点的通量与 k 热启动求解新截面下的特征值（见 README"燃耗"）。
"""
from __future__ import annotations

import copy
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .depletion import (
    deplete_step,
    depleted_mesh,
    initial_remaining,
    region_remaining,
)
from .discretization import BoundaryCondition, build_mesh, region_from_dict
from .solver import (
    ConvergenceError,
    SolverError,
    SolverSettings,
    solve,
)
from .storage import Storage, new_id, utc_now

MAX_BURNUP_STEPS = 500


class BurnupStateError(RuntimeError):
    """历程当前状态不允许该操作（如运行中重复推进、完成后还推进）。"""


def _snapshot_from_version(ver: dict) -> dict:
    return {
        "version": ver["version"],
        "regions": copy.deepcopy(ver["regions"]),
        "left_boundary": ver["left_boundary"],
        "right_boundary": ver["right_boundary"],
        "settings": copy.deepcopy(ver["settings"]),
        "target_total_fission": ver["target_total_fission"],
    }


def _settings(snap: dict) -> SolverSettings:
    s = snap["settings"]
    return SolverSettings(
        k_tol=float(s["k_tol"]),
        flux_tol=float(s["flux_tol"]),
        max_iter=int(s["max_iter"]),
    )


def _check_steps(steps: list[float]) -> None:
    if not (1 <= len(steps) <= MAX_BURNUP_STEPS):
        raise ValueError(
            f"时间步数 {len(steps)} 超出允许范围 1..{MAX_BURNUP_STEPS}")
    for i, dt in enumerate(steps):
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError(
                f"第 {i} 个时间步长必须为正有限值（天），收到 {dt!r}")


def point_to_doc(hist: dict, *, step_index: int, time_days: float,
                 remaining: np.ndarray, region_rem: list[float], res) -> dict:
    """单个时间点的完整留档：k、通量、剩余份额、反应率、两端泄漏、平衡残差。"""
    return {
        "burnup_id": hist["id"],
        "case_id": hist["case_id"],
        "version": hist["version"],
        "step_index": int(step_index),
        "time_days": float(time_days),
        "k_eff": float(res.k_eff),
        "iterations": int(res.iterations),
        "k_residual": float(res.k_residual),
        "flux_residual": float(res.flux_residual),
        "region_remaining": [float(x) for x in region_rem],
        "cell_remaining": np.asarray(remaining, dtype=float).tolist(),
        "region_absorption": [float(x) for x in res.region_absorption],
        "region_fission": [float(x) for x in res.region_fission],
        "leakage_left": float(res.leakage_left),
        "leakage_right": float(res.leakage_right),
        "total_absorption": float(res.total_absorption),
        "total_fission": float(res.total_fission),
        "balance_residual": float(res.balance_residual),
        "target_total_fission": float(res.total_fission),
        "n_mesh_total": int(res.phi.size),
        "centers": np.asarray(res.centers, dtype=float).tolist(),
        "widths": np.asarray(res.widths, dtype=float).tolist(),
        "region_of": np.asarray(res.region_of, dtype=int).tolist(),
        "phi": np.asarray(res.phi, dtype=float).tolist(),
    }


class BurnupManager:
    def __init__(self, storage: Storage, max_workers: int = 4,
                 step_delay: float = 0.0):
        """step_delay：每完成一步后可被取消立即打断的等待秒数（测试钩子，
        生产为 0），与 JobManager 的 eval_delay 同款用途。"""
        self.storage = storage
        self.step_delay = float(step_delay)
        self.pool = ThreadPoolExecutor(max_workers=max_workers,
                                       thread_name_prefix="burnup")
        self._cancel_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        for bid in storage.recover_interrupted_burnups():
            ev = threading.Event()
            ev.set()
            self._cancel_events[bid] = ev

    # ---------- 提交 / 续算 / 取消 ----------

    def create(self, case_id: str, req: dict) -> dict:
        case = self.storage.get_case(case_id)  # 不存在抛 NotFound
        version_no = req.get("version") or case["current_version"]
        ver = self.storage.get_version(case_id, int(version_no))
        steps = [float(x) for x in req["steps"]]
        _check_steps(steps)
        max_steps = req.get("max_steps")
        if max_steps is not None:
            max_steps = int(max_steps)
            if max_steps < 1:
                raise ValueError("max_steps 必须 ≥ 1")

        burnup_id = new_id("bup")
        now = utc_now()
        doc = {
            "id": burnup_id,
            "case_id": case_id,
            "case_name": case["name"],
            "version": ver["version"],
            "status": "running",
            "steps": steps,
            "steps_completed": 0,
            "current_time_days": 0.0,
            "current_k": None,
            "cancelled": False,
            "message": "已提交，后台推进中",
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "note": req.get("note"),
            "snapshot": _snapshot_from_version(ver),
            "point_ids": [],
        }
        self.storage.save_burnup(doc)
        self._submit(burnup_id, max_steps)
        return self.storage.get_burnup(burnup_id)

    def advance(self, burnup_id: str, max_steps: Optional[int] = None) -> dict:
        """接着往后推：paused / cancelled / interrupted / failed 都可续算。"""
        with self._submit_lock:
            hist = self.storage.get_burnup(burnup_id)
            if hist["status"] == "running":
                raise BurnupStateError("历程正在推进中，不能重复启动")
            if hist["status"] == "completed":
                raise BurnupStateError("历程已完成全部时间步，无需继续推进")
            remaining = len(hist["steps"]) - hist["steps_completed"]
            if remaining <= 0:
                raise BurnupStateError("历程没有可推进的剩余步")
            if max_steps is not None:
                max_steps = int(max_steps)
                if max_steps < 1:
                    raise ValueError("max_steps 必须 ≥ 1")
            self.storage.update_burnup(
                burnup_id, status="running", finished_at=None,
                message=f"继续推进剩余 {remaining} 步")
            self._submit(burnup_id, max_steps)
        return self.storage.get_burnup(burnup_id)

    def _submit(self, burnup_id: str, max_steps: Optional[int]) -> None:
        with self._events_lock:
            self._cancel_events[burnup_id] = threading.Event()
        self.pool.submit(self._run, burnup_id, max_steps)

    def cancel(self, burnup_id: str) -> dict:
        hist = self.storage.get_burnup(burnup_id)
        if hist["status"] == "running":
            with self._events_lock:
                ev = self._cancel_events.get(burnup_id)
            if ev is not None:
                ev.set()
            self.storage.update_burnup(
                burnup_id, message="收到取消请求，当前步完成后停下")
        return self.storage.get_burnup(burnup_id)

    def _is_cancelled(self, burnup_id: str) -> bool:
        with self._events_lock:
            ev = self._cancel_events.get(burnup_id)
        return ev is not None and ev.is_set()

    # ---------- 工作线程 ----------

    def _run(self, burnup_id: str, max_steps: Optional[int]) -> None:
        try:
            self._execute(burnup_id, max_steps)
        except Exception as exc:  # noqa: BLE001 - 后台推进任何异常都要落档
            hist = self.storage.get_burnup(burnup_id)
            if hist["status"] == "running":
                self.storage.update_burnup(
                    burnup_id, status="failed", finished_at=utc_now(),
                    message=f"历程内部错误: {type(exc).__name__}: {exc}",
                )

    def _execute(self, burnup_id: str, max_steps: Optional[int]) -> None:
        hist = self.storage.get_burnup(burnup_id)
        if hist["started_at"] is None:
            self.storage.update_burnup(burnup_id, started_at=utc_now())

        snap = hist["snapshot"]
        regions = [region_from_dict(r) for r in snap["regions"]]
        mesh0 = build_mesh(regions)
        left = BoundaryCondition(snap["left_boundary"])
        right = BoundaryCondition(snap["right_boundary"])
        settings = _settings(snap)
        target = float(snap["target_total_fission"])
        steps = hist["steps"]

        points = self.storage.list_burnup_points(burnup_id)
        if points:
            # 续算：从最后一个已落盘点的状态原样恢复（JSON 双精度往返逐位不变）
            last = points[-1]
            remaining = np.asarray(last["cell_remaining"], dtype=float)
            phi = np.asarray(last["phi"], dtype=float)
            k = float(last["k_eff"])
            t = float(last["time_days"])
            done = int(last["step_index"])
        else:
            # 时间点 0：新装料的初始成分（s=1），冷启动求解
            remaining = initial_remaining(mesh0)
            try:
                res = solve(mesh0, left, right, settings, mode="cold",
                            target_total_fission=target)
            except (SolverError, ConvergenceError) as exc:
                self._finish(burnup_id, "failed", f"初始点求解失败: {exc}")
                return
            t, done = 0.0, 0
            self._store_point(hist, regions, mesh0, res, remaining, done, t)
            phi, k = res.phi, res.k_eff
            if self._wait_or_cancel(burnup_id):
                return

        ran = 0
        while done < len(steps):
            if max_steps is not None and ran >= max_steps:
                break
            if self._is_cancelled(burnup_id):
                self._finish(burnup_id, "cancelled",
                             "已取消：已完成的点保留，未完成的步不再写入",
                             cancelled=True)
                return
            dt = float(steps[done])
            # 步首冻结通量解析耗散 → 新截面 → 热启动求解该时刻特征值
            remaining = deplete_step(mesh0, regions, remaining, phi, dt)
            mesh = depleted_mesh(mesh0, regions, remaining)
            try:
                res = solve(mesh, left, right, settings, mode="hot",
                            previous_phi=phi, initial_k=k,
                            target_total_fission=target)
            except (SolverError, ConvergenceError) as exc:
                self._finish(burnup_id, "failed",
                             f"第 {done + 1} 步（t={t + dt:.6g} 天）求解失败: {exc}")
                return
            t += dt
            done += 1
            self._store_point(hist, regions, mesh0, res, remaining, done, t)
            phi, k = res.phi, res.k_eff
            ran += 1
            if self._wait_or_cancel(burnup_id):
                return

        if done >= len(steps):
            self._finish(burnup_id, "completed",
                         f"全部 {len(steps)} 步推进完成")
        else:
            self._finish(burnup_id, "paused",
                         f"已推进 {done}/{len(steps)} 步，可继续推进")

    def _store_point(self, hist: dict, regions, mesh0, res,
                     remaining: np.ndarray, step_index: int, t: float) -> None:
        rr = region_remaining(mesh0, regions, remaining)
        doc = point_to_doc(hist, step_index=step_index, time_days=t,
                           remaining=remaining, region_rem=rr, res=res)
        self.storage.add_burnup_point(
            hist["id"], doc,
            progress={"steps_completed": int(step_index),
                      "current_time_days": float(t),
                      "current_k": float(res.k_eff)})

    def _wait_or_cancel(self, burnup_id: str) -> bool:
        """测试钩子等待（可被取消立即打断）后统一检查取消标志。"""
        if self.step_delay > 0.0:
            with self._events_lock:
                ev = self._cancel_events.get(burnup_id)
            if ev is not None:
                ev.wait(timeout=self.step_delay)
        if self._is_cancelled(burnup_id):
            self._finish(burnup_id, "cancelled",
                         "已取消：已完成的点保留，未完成的步不再写入",
                         cancelled=True)
            return True
        return False

    def _finish(self, burnup_id: str, status: str, message: str, *,
                cancelled: bool = False) -> None:
        fields = {"status": status, "message": message,
                  "finished_at": utc_now()}
        if cancelled:
            fields["cancelled"] = True
        self.storage.update_burnup(burnup_id, **fields)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
