"""燃耗历程的后台推进调度。

- 一条燃耗历程挂在工况某一版参数上：提交时把该版参数快照内嵌进历程文档，
  之后工况被改、被删都不影响历程（与临界搜索作业同一套版本锁定哲学）。
- 固定大小线程池后台执行；每个时间点算完立即落盘（先写点文件、再更新
  历程计数），进度可查（已完成步数/当前燃耗时间/当前 k），可中途取消。
- 分段推进：停下、被取消、甚至服务重启之后，都能用 advance 从最后一个
  已完成时间点继续。续推状态（逐单元剩余份额、通量、k）全部从该点的
  存档精确恢复，因此分段结果与一次性推到底在收敛容差量级内一致。
- 服务重启时仍在 queued/running 的历程标为 interrupted（不自动重跑），
  已完成时间点保留。
"""
from __future__ import annotations

import copy
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from .burnup import (
    FuelExhaustedError,
    depletion_step,
    parse_burnup_params,
    region_remaining,
)
from .discretization import BoundaryCondition, Region, build_mesh
from .solver import ConvergenceError, SolverError, SolverSettings, solve
from .storage import Storage, new_id, utc_now

MAX_TOTAL_STEPS = 500          # 一条历程累计步数上限（与单次请求上限一致）


class BurnupConflictError(RuntimeError):
    """历程正在推进中，拒绝并发的续推/重复提交。"""


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


def check_steps(steps, already_completed: int) -> list[float]:
    """校验一段时间步序列；非法时抛 ValueError（路由层转成 422 并指出 steps）。"""
    if not steps:
        raise ValueError("steps 至少需要一个时间步")
    out = []
    for s in steps:
        v = float(s)
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(f"时间步长度必须为正有限值（天），收到 {s!r}")
        out.append(v)
    if len(out) > MAX_TOTAL_STEPS:
        raise ValueError(f"单次步数 {len(out)} 超过上限 {MAX_TOTAL_STEPS}")
    if already_completed + len(out) > MAX_TOTAL_STEPS:
        raise ValueError(
            f"历程累计步数将达 {already_completed + len(out)}，"
            f"超过上限 {MAX_TOTAL_STEPS}（已完成 {already_completed} 步）")
    return out


class BurnupManager:
    def __init__(self, storage: Storage, max_workers: int = 4,
                 step_delay: float = 0.0):
        """step_delay：每完成一步后额外等待的秒数（测试钩子，生产为 0）。

        等待可被取消事件立即打断，因此既能稳定制造"推进中"的窗口，
        取消本身又不必等满一个 delay。
        """
        self.storage = storage
        self.step_delay = float(step_delay)
        self.pool = ThreadPoolExecutor(max_workers=max_workers,
                                       thread_name_prefix="burnup")
        self._cancel_events: dict[str, threading.Event] = {}
        self._worker_done: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        for bid in storage.recover_interrupted_burnups():
            ev = threading.Event()
            ev.set()
            self._cancel_events[bid] = ev

    # ---------- 提交与续推 ----------

    def submit(self, case_id: str, req: dict) -> dict:
        case = self.storage.get_case(case_id)  # 不存在抛 NotFound
        version = req.get("version") or case["current_version"]
        ver = self.storage.get_version(case_id, int(version))
        steps = check_steps(req["steps"], already_completed=0)

        burnup_id = new_id("burn")
        now = utc_now()
        doc = {
            "id": burnup_id,
            "case_id": case_id,
            "case_name": case["name"],
            "version": ver["version"],
            "status": "queued",
            "steps_completed": 0,
            "steps_planned": len(steps),
            "n_points": 0,
            "burnup_time_days": 0.0,
            "current_k": None,
            "cancelled": False,
            "message": "排队中",
            "note": req.get("note"),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "snapshot": _snapshot_from_version(ver),
        }
        self.storage.create_burnup(doc)
        self._start_worker(burnup_id, steps)
        return self.storage.get_burnup(burnup_id)

    def advance(self, burnup_id: str, req: dict) -> dict:
        """从最后一个已完成时间点接着往后推一段。"""
        with self._submit_lock:
            doc = self.storage.get_burnup(burnup_id)
            if doc["status"] in ("queued", "running"):
                raise BurnupConflictError(
                    f"历程 {burnup_id} 正在推进中，等它停下后再续推")
            # 上一段的工作线程可能刚收到取消、尚未退出：等它干净退出，
            # 避免旧线程的状态写与新段混淆
            done = self._worker_done.get(burnup_id)
            if done is not None and not done.wait(timeout=60.0):
                raise BurnupConflictError(
                    f"历程 {burnup_id} 的上一段推进尚未完全停止")
            steps = check_steps(req["steps"],
                                already_completed=doc["steps_completed"])
            self.storage.update_burnup(
                burnup_id,
                status="queued",
                steps_planned=len(steps),
                cancelled=False,
                finished_at=None,
                message="排队中（续推）",
            )
            self._start_worker(burnup_id, steps)
            return self.storage.get_burnup(burnup_id)

    def _start_worker(self, burnup_id: str, steps: list[float]) -> None:
        with self._events_lock:
            self._cancel_events[burnup_id] = threading.Event()
            self._worker_done[burnup_id] = threading.Event()
        self.pool.submit(self._run, burnup_id, steps)

    def cancel(self, burnup_id: str) -> dict:
        doc = self.storage.get_burnup(burnup_id)
        if doc["status"] in ("queued", "running"):
            with self._events_lock:
                ev = self._cancel_events.get(burnup_id)
            if ev is not None:
                ev.set()
            self.storage.update_burnup(
                burnup_id,
                message="收到取消请求，等待当前步结束" if doc["status"] == "running"
                        else "已取消（排队中）",
            )
        return self.storage.get_burnup(burnup_id)

    def _is_cancelled(self, burnup_id: str) -> bool:
        with self._events_lock:
            ev = self._cancel_events.get(burnup_id)
        return ev is not None and ev.is_set()

    # ---------- 工作线程 ----------

    def _run(self, burnup_id: str, steps: list[float]) -> None:
        try:
            self._execute(burnup_id, steps)
        except Exception as exc:  # noqa: BLE001 - 后台推进任何异常都要落档
            doc = self.storage.get_burnup(burnup_id)
            if doc["status"] in ("queued", "running"):
                self.storage.update_burnup(
                    burnup_id,
                    status="failed",
                    finished_at=utc_now(),
                    message=f"推进内部错误: {type(exc).__name__}: {exc}",
                )
        finally:
            with self._events_lock:
                done = self._worker_done.get(burnup_id)
            if done is not None:
                done.set()

    def _execute(self, burnup_id: str, steps: list[float]) -> None:
        doc = self.storage.get_burnup(burnup_id)
        if self._is_cancelled(burnup_id):
            self._finish_cancelled(burnup_id)
            return

        self.storage.update_burnup(
            burnup_id, status="running",
            started_at=doc["started_at"] or utc_now(),
            message="推进中")

        snap = doc["snapshot"]
        regions = [Region(**r) for r in snap["regions"]]
        mesh = build_mesh(regions)
        params = parse_burnup_params(snap["regions"])
        left = BoundaryCondition(snap["left_boundary"])
        right = BoundaryCondition(snap["right_boundary"])
        settings = _settings(snap)
        target = float(snap["target_total_fission"])

        n_points = doc["n_points"]
        if n_points == 0:
            # 新装料：t=0 初始点（冷启动），r ≡ 1
            remaining = np.ones(mesh.n, dtype=float)
            time_days = 0.0
            res = solve(mesh, left, right, settings, mode="cold",
                        target_total_fission=target)
            if self._is_cancelled(burnup_id):
                self._finish_cancelled(burnup_id)
                return
            self._write_point(doc, mesh, 0, remaining, 0.0, 0.0, res)
            n_points = 1
            self.storage.update_burnup(
                burnup_id, n_points=n_points, current_k=res.k_eff,
                burnup_time_days=0.0,
                message=f"初始点已写入，推进中（0/{len(steps)}）")
            phi_prev, k_prev = res.phi, res.k_eff
        else:
            # 续推：从最后一个已完成时间点精确恢复状态
            last = self.storage.get_burnup_point(burnup_id, n_points - 1)
            remaining = np.asarray(last["cell_remaining"], dtype=float)
            time_days = float(last["time_days"])
            phi_prev = np.asarray(last["phi"], dtype=float)
            k_prev = float(last["k_eff"])

        completed_here = 0
        for step_days in steps:
            if self._is_cancelled(burnup_id):
                self._finish_cancelled(burnup_id)
                return
            try:
                remaining, res = depletion_step(
                    mesh, left, right, settings, params,
                    remaining, phi_prev, k_prev, step_days, target)
            except FuelExhaustedError as exc:
                # 燃料烧尽：已完成的点全部保留，历程以 exhausted 终态结束
                self.storage.update_burnup(
                    burnup_id, status="exhausted", finished_at=utc_now(),
                    message=(f"燃料耗尽（{exc}）：推进在第 "
                             f"{doc['steps_completed'] + completed_here + 1} 步"
                             "前停止，已完成的时间点保留"))
                return
            time_days += step_days
            if self._is_cancelled(burnup_id):
                # 未完成的步不写入；已完成的点全部保留
                self._finish_cancelled(burnup_id)
                return
            self._write_point(doc, mesh, n_points, remaining,
                              time_days, step_days, res)
            n_points += 1
            completed_here += 1
            phi_prev, k_prev = res.phi, res.k_eff
            self.storage.update_burnup(
                burnup_id,
                steps_completed=doc["steps_completed"] + completed_here,
                n_points=n_points,
                burnup_time_days=time_days,
                current_k=res.k_eff,
                message=f"推进中（{completed_here}/{len(steps)}）",
            )
            if self.step_delay > 0.0:
                with self._events_lock:
                    ev = self._cancel_events.get(burnup_id)
                if ev is not None:
                    ev.wait(timeout=self.step_delay)

        self.storage.update_burnup(
            burnup_id, status="completed", finished_at=utc_now(),
            message=f"本段 {len(steps)} 步全部完成，"
                    f"累计 {doc['steps_completed'] + completed_here} 步，"
                    f"燃耗时间 {time_days:g} 天")

    def _finish_cancelled(self, burnup_id: str) -> None:
        self.storage.update_burnup(
            burnup_id, status="cancelled", cancelled=True,
            finished_at=utc_now(),
            message="已取消：已完成的时间点保留，未完成的步不写入",
        )

    def _write_point(self, doc: dict, mesh, index: int,
                     remaining: np.ndarray, time_days: float,
                     step_days: float, res) -> None:
        """落盘一个时间点（k、通量、各区剩余份额、各区反应率、两端泄漏、
        中子平衡残差，以及续推所需的逐单元剩余份额）。"""
        point = {
            "id": f"{doc['id']}_p{index:05d}",
            "burnup_id": doc["id"],
            "case_id": doc["case_id"],
            "case_name": doc["case_name"],
            "version": doc["version"],
            "step_index": index,
            "time_days": float(time_days),
            "step_days": float(step_days),
            "k_eff": float(res.k_eff),
            "region_remaining": region_remaining(mesh, remaining),
            "cell_remaining": np.asarray(remaining, dtype=float).tolist(),
            "region_absorption": [float(x) for x in res.region_absorption],
            "region_fission": [float(x) for x in res.region_fission],
            "leakage_left": float(res.leakage_left),
            "leakage_right": float(res.leakage_right),
            "total_absorption": float(res.total_absorption),
            "total_fission": float(res.total_fission),
            "balance_residual": float(res.balance_residual),
            "n_mesh_total": int(res.phi.size),
            "centers": np.asarray(res.centers, dtype=float).tolist(),
            "widths": np.asarray(res.widths, dtype=float).tolist(),
            "region_of": np.asarray(res.region_of, dtype=int).tolist(),
            "phi": np.asarray(res.phi, dtype=float).tolist(),
            "created_at": utc_now(),
        }
        self.storage.save_burnup_point(doc["id"], index, point)

    def shutdown(self) -> None:
        # 置上所有取消标志，让在跑的线程尽快干净退出（终态 cancelled）
        with self._events_lock:
            events = list(self._cancel_events.values())
        for ev in events:
            ev.set()
        self.pool.shutdown(wait=False, cancel_futures=True)
