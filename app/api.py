"""FastAPI 路由层：工况 CRUD、求解、结果留档、临界搜索作业、燃耗历程。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .burnup_jobs import BurnupConflictError, BurnupManager
from .discretization import BoundaryCondition, Region, build_mesh
from .interpolation import piecewise_linear_remap
from .jobs import JobManager, result_to_doc
from .models import (
    BurnupAdvance,
    BurnupCreate,
    BurnupOut,
    BurnupPointOut,
    BurnupPointSummary,
    CaseCreate,
    CaseOut,
    CasePatch,
    CaseSummary,
    JobOut,
    ResultOut,
    SolveRequest,
    SearchRequest,
)
from .solver import (
    ConvergenceError,
    SolverError,
    SolverSettings,
    solve,
)
from .storage import NotFound, Storage
from .validation import validate_payload


def create_app(storage: Storage, eval_delay: float = 0.0,
               burnup_step_delay: float = 0.0) -> FastAPI:
    app = FastAPI(
        title="一维平板单群扩散特征值服务",
        version="1.1.0",
        description="多区平板 k_eff 求解、版本化工况、异步临界搜索与燃耗历程",
    )
    jobs = JobManager(storage, eval_delay=eval_delay)
    burnups = BurnupManager(storage, step_delay=burnup_step_delay)
    app.state.storage = storage
    app.state.jobs = jobs
    app.state.burnups = burnups

    # ---------- 异常 ----------

    @app.exception_handler(NotFound)
    async def not_found_handler(_request, exc: NotFound):
        return JSONResponse(status_code=404,
                            content={"error": str(exc), "details": []})

    @app.exception_handler(ValueError)
    async def value_error_handler(_request, exc: ValueError):
        return JSONResponse(status_code=400,
                            content={"error": str(exc), "details": []})

    @app.exception_handler(BurnupConflictError)
    async def burnup_conflict_handler(_request, exc: BurnupConflictError):
        return JSONResponse(status_code=409,
                            content={"error": str(exc), "details": []})

    # ---------- 工具 ----------

    def _check_regions(regions_data: list[dict]) -> None:
        errs = validate_payload({"regions": regions_data})
        if errs:
            raise HTTPException(status_code=422, detail={
                "error": "工况物理校验未通过",
                "details": [{"field": f, "message": m} for f, m in errs],
            })

    def _case_out(case: dict) -> dict:
        cur = next(v for v in case["versions"]
                   if v["version"] == case["current_version"])
        return {
            "id": case["id"],
            "name": case["name"],
            "current_version": case["current_version"],
            "created_at": case["created_at"],
            "updated_at": case["updated_at"],
            "regions": cur["regions"],
            "left_boundary": cur["left_boundary"],
            "right_boundary": cur["right_boundary"],
            "settings": cur["settings"],
            "target_total_fission": cur["target_total_fission"],
            "result_ids": cur["result_ids"],
        }

    def _summary(case: dict) -> dict:
        return {
            "id": case["id"],
            "name": case["name"],
            "current_version": case["current_version"],
            "created_at": case["created_at"],
            "updated_at": case["updated_at"],
        }

    def _settings_for(ver: dict, override: Optional[dict]) -> SolverSettings:
        merged = dict(ver["settings"])
        if override:
            for key in ("k_tol", "flux_tol", "max_iter"):
                val = override.get(key)
                if val is not None:
                    merged[key] = val
        return SolverSettings(k_tol=float(merged["k_tol"]),
                              flux_tol=float(merged["flux_tol"]),
                              max_iter=int(merged["max_iter"]))

    def _result_out(doc: dict) -> dict:
        return doc

    # ---------- 健康检查 ----------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ---------- 工况 ----------

    @app.post("/cases", response_model=CaseOut, status_code=201)
    def create_case(body: CaseCreate):
        # exclude_none：未声明燃耗的区落盘格式与燃耗功能引入前完全一致
        regions_data = [r.model_dump(exclude_none=True) for r in body.regions]
        _check_regions(regions_data)
        case = storage.create_case({
            "name": body.name,
            "regions": regions_data,
            "left_boundary": body.left_boundary,
            "right_boundary": body.right_boundary,
            "settings": body.settings.model_dump(exclude_none=True),
            "target_total_fission": body.target_total_fission,
        })
        return _case_out(case)

    @app.get("/cases", response_model=list[CaseSummary])
    def list_cases():
        return [_summary(c) for c in storage.list_cases()]

    @app.get("/cases/{case_id}", response_model=CaseOut)
    def get_case(case_id: str):
        return _case_out(storage.get_case(case_id))

    @app.patch("/cases/{case_id}", response_model=CaseOut)
    def patch_case(case_id: str, body: CasePatch):
        current = storage.get_case(case_id)
        cur_ver = next(v for v in current["versions"]
                       if v["version"] == current["current_version"])

        patch = body.model_dump(exclude_none=True)
        if "region" in patch and "region_index" not in patch:
            raise HTTPException(status_code=422, detail={
                "error": "修改区参数时必须同时给出 region_index",
                "details": [{"field": "region_index", "message": "缺失"}],
            })
        if "region_index" in patch and "region" not in patch:
            # 只给下标不给补丁：无意义但无害，直接忽略
            patch.pop("region_index")

        # 先在内存里拼出候选版本并做物理校验，通过后才落盘（不产生非法版本）
        candidate_regions = [dict(r) for r in cur_ver["regions"]]
        if "region" in patch:
            idx = patch["region_index"]
            if not (0 <= idx < len(candidate_regions)):
                raise HTTPException(status_code=404, detail={
                    "error": f"region_index={idx} 越界"
                             f"（共 {len(candidate_regions)} 区）",
                    "details": [],
                })
            for key, val in patch["region"].items():
                candidate_regions[idx][key] = val
        errs = validate_payload({"regions": candidate_regions})
        if errs:
            raise HTTPException(status_code=422, detail={
                "error": "修改后工况物理校验未通过，已拒绝（未生成新版本）",
                "details": [{"field": f, "message": m} for f, m in errs],
            })

        try:
            case = storage.patch_case(case_id, patch)
        except NotFound:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={
                "error": str(exc), "details": []})
        return _case_out(case)

    @app.delete("/cases/{case_id}", status_code=204)
    def delete_case(case_id: str):
        storage.delete_case(case_id)
        return JSONResponse(status_code=204, content=None)

    @app.get("/cases/{case_id}/versions")
    def list_versions(case_id: str):
        case = storage.get_case(case_id)
        return [{
            "version": v["version"],
            "created_at": v["created_at"],
            "change_summary": v["change_summary"],
            "regions": v["regions"],
            "left_boundary": v["left_boundary"],
            "right_boundary": v["right_boundary"],
            "settings": v["settings"],
            "target_total_fission": v["target_total_fission"],
            "result_ids": v["result_ids"],
        } for v in case["versions"]]

    # ---------- 求解 ----------

    @app.post("/cases/{case_id}/solve", response_model=ResultOut)
    def solve_case(case_id: str, body: SolveRequest):
        case = storage.get_case(case_id)
        ver = storage.get_version(case_id, case["current_version"])

        warm_source: Optional[dict] = None
        warm_result_id: Optional[str] = None
        if body.mode == "hot":
            # 优先上一版的最新收敛结果；同版已有结果时也允许用来续算
            if case["current_version"] >= 2:
                warm_source = storage.latest_converged_result(
                    case_id, case["current_version"] - 1)
            if warm_source is None:
                warm_source = storage.latest_converged_result(
                    case_id, case["current_version"])
            if warm_source is None:
                raise HTTPException(status_code=409, detail={
                    "error": "没有可用于热启动的历史收敛结果，"
                             "请先对该工况（或其上一版）做一次冷启动求解",
                    "details": [],
                })
            warm_result_id = warm_source["id"]

        regions = [Region(**r) for r in ver["regions"]]
        mesh = build_mesh(regions)
        left = BoundaryCondition(ver["left_boundary"])
        right = BoundaryCondition(ver["right_boundary"])
        settings = _settings_for(ver, body.settings.model_dump(exclude_none=True)
                                 if body.settings else None)
        target = (body.target_total_fission
                  if body.target_total_fission is not None
                  else ver["target_total_fission"])

        prev_phi = None
        if body.mode == "hot":
            prev_phi = piecewise_linear_remap(
                np.asarray(warm_source["centers"], dtype=float),
                np.asarray(warm_source["phi"], dtype=float),
                mesh.centers,
            )

        try:
            res = solve(mesh, left, right, settings,
                        mode=body.mode, previous_phi=prev_phi,
                        initial_k=(warm_source["k_eff"]
                                   if body.mode == "hot" else None),
                        target_total_fission=float(target))
        except ConvergenceError as exc:
            raise HTTPException(status_code=409, detail={
                "error": str(exc),
                "ok": False,
                "iterations": exc.iterations,
                "k": exc.k,
                "k_residual": exc.k_residual,
                "flux_residual": exc.flux_residual,
            })
        except SolverError as exc:
            raise HTTPException(status_code=400,
                                detail={"error": f"求解失败: {exc}", "details": []})

        doc = storage.add_result(case_id, ver["version"], result_to_doc(
            res, note=body.note, warm_from_result=warm_result_id))
        return _result_out(doc)

    @app.get("/cases/{case_id}/results", response_model=list[ResultOut])
    def list_case_results(case_id: str,
                          version: Optional[int] = Query(default=None)):
        return storage.list_results(case_id, version)

    @app.get("/results/{result_id}", response_model=ResultOut)
    def get_result(result_id: str):
        return storage.get_result(result_id)

    # ---------- 临界搜索作业 ----------

    @app.post("/cases/{case_id}/searches", response_model=JobOut, status_code=202)
    def submit_search(case_id: str, body: SearchRequest):
        # 作业端点校核需要先保证工况存在；具体同号判定在后台首步完成
        try:
            job = jobs.submit(case_id, body.model_dump())
        except NotFound:
            raise
        except KeyError as exc:
            raise HTTPException(status_code=404,
                                detail={"error": str(exc), "details": []})
        except ValueError as exc:
            raise HTTPException(status_code=422,
                                detail={"error": str(exc), "details": []})
        return job

    @app.get("/jobs", response_model=list[JobOut])
    def list_jobs(case_id: Optional[str] = Query(default=None)):
        return storage.list_jobs(case_id)

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str):
        return storage.get_job(job_id)

    @app.post("/jobs/{job_id}/cancel", response_model=JobOut)
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    # ---------- 燃耗历程 ----------

    def _steps_422(exc: ValueError) -> HTTPException:
        return HTTPException(status_code=422, detail={
            "error": str(exc),
            "details": [{"field": "steps", "message": str(exc)}],
        })

    @app.post("/cases/{case_id}/burnups", response_model=BurnupOut,
              status_code=202)
    def create_burnup(case_id: str, body: BurnupCreate):
        """新建燃耗历程并后台推进。锁定指定（缺省当前）参数版本的快照。"""
        try:
            return burnups.submit(case_id, body.model_dump())
        except NotFound:
            raise
        except ValueError as exc:
            raise _steps_422(exc)

    @app.get("/cases/{case_id}/burnups", response_model=list[BurnupOut])
    def list_case_burnups(case_id: str):
        storage.get_case(case_id)  # 不存在抛 404
        return storage.list_burnups(case_id)

    @app.get("/burnups/{burnup_id}", response_model=BurnupOut)
    def get_burnup(burnup_id: str):
        return storage.get_burnup(burnup_id)

    @app.post("/burnups/{burnup_id}/advance", response_model=BurnupOut,
              status_code=202)
    def advance_burnup(burnup_id: str, body: BurnupAdvance):
        """从最后一个已完成时间点接着往后推一段（分段推进）。"""
        try:
            return burnups.advance(burnup_id, body.model_dump())
        except NotFound:
            raise
        except BurnupConflictError:
            raise
        except ValueError as exc:
            raise _steps_422(exc)

    @app.post("/burnups/{burnup_id}/cancel", response_model=BurnupOut)
    def cancel_burnup(burnup_id: str):
        return burnups.cancel(burnup_id)

    @app.get("/burnups/{burnup_id}/points",
             response_model=list[BurnupPointSummary])
    def list_burnup_points(burnup_id: str):
        return storage.list_burnup_points(burnup_id)

    @app.get("/burnups/{burnup_id}/points/{index}",
             response_model=BurnupPointOut)
    def get_burnup_point(burnup_id: str, index: int):
        if index < 0:
            raise HTTPException(status_code=422, detail={
                "error": "时间点下标必须 ≥ 0",
                "details": [{"field": "index", "message": "必须 ≥ 0"}],
            })
        return storage.get_burnup_point(burnup_id, index)

    return app
