"""异步临界搜索作业调度。

- 固定大小线程池后台执行；提交即返回作业号。
- 作业文档内嵌提交时刻的参数版本**快照**（regions/边界/设置/目标归一），
  运行中工况被改、被删都不影响作业，结果只挂在被锁定的版本下。
- 每个作业自带 threading.Event 取消标志；每一步二分前后都检查，
  取消后线程立即退出，不再写结果（终态 cancelled 除外）。
- 起点先单独求区间两端 k：两端 k−1 同号立即以 rejected 结束并回报两端 k；
  异号才进入手写二分，成功要求 |k−1| < 1e-6。
- 步数用完未达标 → unconverged，回报最后区间与残差。
"""
from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .discretization import BoundaryCondition, build_mesh, region_from_dict
from .interpolation import piecewise_linear_remap
from .rootfinding import bisection, same_sign
from .solver import (
    SolverSettings,
    ConvergenceError,
    SolverError,
    solve,
)
from .storage import Storage, new_id, utc_now

K_TARGET_TOL = 1e-6          # 临界判据：代回必须 |k−1| < 1e-6
X_TOL = 1e-12


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


def _build_mesh(snap: dict, regions_data: list[dict]):
    regions = [region_from_dict(r) for r in regions_data]
    return build_mesh(regions), regions


class JobManager:
    def __init__(self, storage: Storage, max_workers: int = 4,
                 eval_delay: float = 0.0):
        """eval_delay：每次 k 评估后额外等待的秒数（测试钩子，生产为 0）。

        等待可被取消事件立即打断，因此既能稳定制造"作业在跑"的窗口，
        取消本身又不必等满一个 delay。
        """
        self.storage = storage
        self.eval_delay = float(eval_delay)
        self.pool = ThreadPoolExecutor(max_workers=max_workers,
                                       thread_name_prefix="crit-search")
        self._cancel_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        for jid in storage.recover_interrupted_jobs():
            ev = threading.Event()
            ev.set()
            self._cancel_events[jid] = ev

    # ---------- 提交 ----------

    def submit(self, case_id: str, req: dict) -> dict:
        case = self.storage.get_case(case_id)  # 不存在抛 NotFound
        ver = self.storage.get_version(case_id, case["current_version"])
        req_region_index = int(req["region_index"])
        if not (0 <= req_region_index < len(ver["regions"])):
            raise KeyError(
                f"region_index={req_region_index} 越界（共 {len(ver['regions'])} 区）")
        lower, upper = float(req["lower"]), float(req["upper"])
        if not (lower < upper):
            raise ValueError("搜索区间必须满足 lower < upper，且二者为有限值")
        import math
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError("搜索区间端点必须有限")
        if req["parameter"] == "thickness" and lower <= 0:
            raise ValueError("厚度搜索下界必须 > 0")
        if req["parameter"] == "nu_sigma_f" and lower < 0:
            raise ValueError("νΣf 搜索下界不能为负")

        job_id = new_id("job")
        snapshot = _snapshot_from_version(ver)
        now = utc_now()
        job = {
            "id": job_id,
            "case_id": case_id,
            "case_name": case["name"],
            "version": ver["version"],
            "status": "queued",
            "region_index": req_region_index,
            "parameter": req["parameter"],
            "lower": lower,
            "upper": upper,
            "max_steps": int(req["max_steps"]),
            "steps_taken": 0,
            "current_bracket": None,
            "current_k": None,
            "root_value": None,
            "k_at_root": None,
            "endpoint_k": None,
            "message": "排队中",
            "submitted_at": now,
            "started_at": None,
            "finished_at": None,
            "cancelled": False,
            "result_id": None,
            "note": req.get("note"),
            "snapshot": snapshot,
        }
        self.storage.save_job(job)
        with self._events_lock:
            self._cancel_events[job_id] = threading.Event()
        self.pool.submit(self._run, job_id)
        return self.storage.get_job(job_id)

    def cancel(self, job_id: str) -> dict:
        job = self.storage.get_job(job_id)
        if job["status"] in ("queued", "running"):
            with self._events_lock:
                ev = self._cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            # 终态由工作线程写入；若线程刚好还没起步，这里也先标记意图
            self.storage.update_job(
                job_id,
                message="收到取消请求，等待线程退出" if job["status"] == "running"
                       else "已取消（排队中）",
            )
        return self.storage.get_job(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._events_lock:
            ev = self._cancel_events.get(job_id)
        return ev is not None and ev.is_set()

    # ---------- 工作线程 ----------

    def _run(self, job_id: str) -> None:
        try:
            self._execute(job_id)
        except Exception as exc:  # noqa: BLE001 - 后台作业任何异常都要落档
            job = self.storage.get_job(job_id)
            if job["status"] in ("queued", "running"):
                self.storage.update_job(
                    job_id,
                    status="failed",
                    finished_at=utc_now(),
                    message=f"作业内部错误: {type(exc).__name__}: {exc}",
                )

    def _evaluate_k(self, job: dict, value: float,
                    warm: Optional[dict]) -> tuple[float, dict]:
        """在待定量=value 处冷/热求解一次，返回 (k, 本次结果文档)。

        warm 为上一次收敛结果（含 centers/phi/k）时，重映到本次网格做热启动，
        并继承 k，省步数；端点第一杆没有 warm，走冷启动（k 从 1 起步）。
        """
        snap = job["snapshot"]
        data = copy.deepcopy(snap["regions"])
        data[job["region_index"]][job["parameter"]] = value
        mesh, _ = _build_mesh(snap, data)
        left = BoundaryCondition(snap["left_boundary"])
        right = BoundaryCondition(snap["right_boundary"])

        mode = "cold"
        prev_phi = None
        initial_k = None
        if warm is not None:
            prev_phi = piecewise_linear_remap(
                warm["_centers_arr"], warm["_phi_arr"], mesh.centers)
            mode = "hot"
            initial_k = warm["_k"]

        res = solve(
            mesh, left, right, _settings(snap),
            mode=mode, previous_phi=prev_phi, initial_k=initial_k,
            target_total_fission=float(snap["target_total_fission"]),
        )
        if self.eval_delay > 0.0:
            with self._events_lock:
                ev = self._cancel_events.get(job["id"])
            if ev is not None:
                ev.wait(timeout=self.eval_delay)
        return res.k_eff, res

    def _execute(self, job_id: str) -> None:
        job = self.storage.get_job(job_id)
        if self._is_cancelled(job_id):
            self._finish_cancelled(job_id)
            return

        self.storage.update_job(job_id, status="running",
                                started_at=utc_now(), message="正在求区间端点 k")

        snap = job["snapshot"]
        lower, upper = job["lower"], job["upper"]

        # --- 两端点（第一杆冷启动，第二杆用第一杆热启动） ---
        try:
            k_lo, res_lo = self._evaluate_k(job, lower, warm=None)
            if self._is_cancelled(job_id):
                self._finish_cancelled(job_id)
                return
            k_hi, res_hi = self._evaluate_k(job, upper, warm=_carry(res_lo))
        except (SolverError, ConvergenceError) as exc:
            self.storage.update_job(
                job_id, status="failed", finished_at=utc_now(),
                message=f"端点 k 求解失败: {exc}")
            return

        f_lo, f_hi = k_lo - 1.0, k_hi - 1.0
        self.storage.update_job(
            job_id, endpoint_k=(k_lo, k_hi),
            current_bracket=(lower, upper),
            current_k=None)

        # --- 同号直接拒绝 ---
        if f_lo == 0.0 or f_hi == 0.0 or same_sign(f_lo, f_hi):
            root_value = lower if f_lo == 0.0 else upper
            k_at = 1.0
            status = "converged" if (f_lo == 0.0 or f_hi == 0.0) else "rejected"
            result_id = None
            if status == "converged":
                result_id = self._archive_solution(job, root_value,
                                                   res_lo if f_lo == 0.0 else res_hi)
            self.storage.update_job(
                job_id,
                status=status,
                finished_at=utc_now(),
                root_value=root_value if status == "converged" else None,
                k_at_root=k_at if status == "converged" else None,
                result_id=result_id,
                message=("端点 k−1 同号，区间内无（唯一）临界点，拒绝搜索"
                         if status == "rejected"
                         else "端点恰好临界"),
            )
            return

        # --- 二分 ---
        last_res: Optional[dict] = None
        last_mid: Optional[float] = None

        def func(x: float) -> float:
            nonlocal last_res, last_mid
            warm = _carry(last_res) if last_res is not None else _carry(res_hi)
            k, res = self._evaluate_k(job, x, warm=warm)
            last_mid, last_res = x, res
            return k - 1.0

        def on_progress(step: int, bracket, f_mid: float) -> None:
            self.storage.update_job(
                job_id,
                steps_taken=step,
                current_bracket=(bracket.lo, bracket.hi),
                current_k=1.0 + f_mid,
                message=f"第 {step}/{job['max_steps']} 步二分中",
            )

        result = bisection(
            func, lower, upper, f_lo, f_hi,
            ftol=K_TARGET_TOL, xtol=X_TOL, max_iter=job["max_steps"],
            on_progress=on_progress,
            is_cancelled=lambda: self._is_cancelled(job_id),
        )

        if result.cancelled:
            self._finish_cancelled(job_id)
            return

        if result.converged:
            # root 是最后一个 mid；last_res 即该点完整解
            if last_res is None or abs(last_mid - result.root) > 1e-15:
                # 极小概率命中 xtol 分支，补一杆（用邻近解热启动）
                warm = _carry(last_res) if last_res is not None else None
                _, last_res = self._evaluate_k(job, result.root, warm=warm)
            k_check, final_res = self._evaluate_k(
                job, result.root, warm=_carry(last_res))
            if abs(k_check - 1.0) >= K_TARGET_TOL:
                self.storage.update_job(
                    job_id, status="unconverged", finished_at=utc_now(),
                    steps_taken=result.iterations,
                    root_value=result.root, k_at_root=k_check,
                    current_bracket=(result.bracket.lo, result.bracket.hi),
                    current_k=k_check,
                    message=(f"代回校核 |k−1|={abs(k_check - 1.0):.3e} "
                             f"≥ {K_TARGET_TOL:g}，判为未收敛"),
                )
                return
            result_id = self._archive_solution(job, result.root, final_res)
            self.storage.update_job(
                job_id, status="converged", finished_at=utc_now(),
                steps_taken=result.iterations,
                root_value=result.root, k_at_root=k_check,
                result_id=result_id,
                current_bracket=(result.bracket.lo, result.bracket.hi),
                current_k=k_check,
                message=result.message,
            )
        else:
            self.storage.update_job(
                job_id, status="unconverged", finished_at=utc_now(),
                steps_taken=result.iterations,
                root_value=result.root, k_at_root=1.0 + result.f_root,
                current_bracket=(result.bracket.lo, result.bracket.hi),
                current_k=1.0 + result.f_root,
                message=f"{result.message}；最后区间 "
                        f"[{result.bracket.lo:.8g}, {result.bracket.hi:.8g}]，"
                        f"残差 |k−1|={result.f_root:.3e}",
            )

    def _finish_cancelled(self, job_id: str) -> None:
        self.storage.update_job(
            job_id, status="cancelled", cancelled=True,
            finished_at=utc_now(),
            message="作业已取消，线程退出，不再写入结果",
        )

    def _archive_solution(self, job: dict, value: float, res) -> Optional[str]:
        """把临界点处的完整解归档到被锁定的参数版本。"""
        from .solver import SolveResult  # noqa: F401
        snap = job["snapshot"]
        data = copy.deepcopy(snap["regions"])
        data[job["region_index"]][job["parameter"]] = value
        # 临界点解对应的网格就是 override 后的网格，已在 res 中；
        # 为版本留档，在锁定版本上存结果（不动工况当前版本）
        doc = result_to_doc(res, note=(
            f"临界搜索作业 {job['id']}：第 {job['region_index']} 区 "
            f"{job['parameter']}={value:.10g}"))
        return self.storage.add_result(job["case_id"], snap["version"], doc)["id"]

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def _carry(res) -> dict:
    """把 SolveResult 打包成热启动载体（缓存 numpy 数组与 k）。"""
    return {"_phi_arr": res.phi, "_centers_arr": res.centers, "_k": res.k_eff}


def result_to_doc(res, *, note: Optional[str] = None,
                  warm_from_result: Optional[str] = None) -> dict:
    import numpy as np
    return {
        "mode": res.start_mode,
        "k_eff": float(res.k_eff),
        "iterations": int(res.iterations),
        "k_residual": float(res.k_residual),
        "flux_residual": float(res.flux_residual),
        "target_total_fission": float(res.total_fission),
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
        "note": note,
        "warm_from_result": warm_from_result,
    }
