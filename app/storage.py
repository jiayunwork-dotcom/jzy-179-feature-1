"""JSON 文件持久化：工况 / 参数版本 / 求解结果 / 作业。

- 每类对象一个文件，写入走 "临时文件 + os.replace" 原子替换，
  服务异常退出不会留下半截 JSON。
- 进程内一把可重入锁串行化"改内存对象→落盘"；常驻服务 + 后台线程
  共用同一个 Storage 实例。
- 目录可直接挂卷（DIFFUSION_DATA_DIR，默认 /data）。
- 工况删除只删工况自己的版本与结果索引；作业记录里内嵌参数快照，
  因此作业永远只认提交那一刻的版本，工况被删/被改都不影响在跑的作业。
"""
from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def to_plain(obj: Any) -> Any:
    """把 numpy 标量/数组转成 JSON 可存的原生类型。"""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None  # type: ignore
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


DEFAULT_SETTINGS = {"k_tol": 1e-9, "flux_tol": 1e-9, "max_iter": 100_000}


def resolve_settings(data: Optional[dict]) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if data:
        for key in ("k_tol", "flux_tol", "max_iter"):
            val = data.get(key)
            if val is not None:
                s[key] = val
    return s


class NotFound(KeyError):
    pass


class Storage:
    def __init__(self, data_dir: str | os.PathLike):
        self.root = Path(data_dir)
        self.cases_dir = self.root / "cases"
        self.results_dir = self.root / "results"
        self.jobs_dir = self.root / "jobs"
        for d in (self.cases_dir, self.results_dir, self.jobs_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cases: dict[str, dict] = {}
        self._jobs: dict[str, dict] = {}
        self._load_all()

    # ---------- 底层 ----------

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(to_plain(data), fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _load_all(self) -> None:
        for p in sorted(self.cases_dir.glob("*.json")):
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            self._cases[doc["id"]] = doc
        for p in sorted(self.jobs_dir.glob("*.json")):
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            self._jobs[doc["id"]] = doc

    def _case_path(self, case_id: str) -> Path:
        return self.cases_dir / f"{case_id}.json"

    def _result_path(self, result_id: str) -> Path:
        return self.results_dir / f"{result_id}.json"

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save_case(self, case: dict) -> None:
        self._atomic_write_json(self._case_path(case["id"]), case)

    def _save_job(self, job: dict) -> None:
        self._atomic_write_json(self._job_path(job["id"]), job)

    # ---------- 工况 ----------

    def create_case(self, payload: dict) -> dict:
        with self._lock:
            now = utc_now()
            case_id = new_id("case")
            version = {
                "version": 1,
                "created_at": now,
                "change_summary": "初始版本",
                "regions": copy.deepcopy(payload["regions"]),
                "left_boundary": payload["left_boundary"],
                "right_boundary": payload["right_boundary"],
                "settings": resolve_settings(payload.get("settings")),
                "target_total_fission": payload["target_total_fission"],
                "result_ids": [],
            }
            case = {
                "id": case_id,
                "name": payload["name"],
                "created_at": now,
                "updated_at": now,
                "current_version": 1,
                "versions": [version],
            }
            self._cases[case_id] = case
            self._save_case(case)
            return copy.deepcopy(case)

    def list_cases(self) -> list[dict]:
        with self._lock:
            return [copy.deepcopy(c) for c in
                    sorted(self._cases.values(), key=lambda c: c["created_at"])]

    def get_case(self, case_id: str) -> dict:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise NotFound(f"工况 {case_id} 不存在")
            return copy.deepcopy(case)

    def get_version(self, case_id: str, version: int) -> dict:
        case = self.get_case(case_id)
        for v in case["versions"]:
            if v["version"] == version:
                return copy.deepcopy(v)
        raise NotFound(f"工况 {case_id} 没有版本 {version}")

    def current_version(self, case_id: str) -> dict:
        case = self.get_case(case_id)
        return self.get_version(case_id, case["current_version"])

    def rename_case(self, case_id: str, name: str) -> dict:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise NotFound(f"工况 {case_id} 不存在")
            case["name"] = name
            case["updated_at"] = utc_now()
            self._save_case(case)
            return copy.deepcopy(case)

    def patch_case(self, case_id: str, patch: dict) -> dict:
        """应用修改。除 name 外的任何参数变化都生成一个新版本。

        region 补丁只改 region_index 指定的那一个区，缺省字段保持原值。
        """
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise NotFound(f"工况 {case_id} 不存在")
            cur = case["versions"][-1]

            name = patch.get("name")
            geometry_changed = False

            new_regions = copy.deepcopy(cur["regions"])
            region_patch = patch.get("region")
            region_index = patch.get("region_index")
            if region_patch:
                if region_index is None:
                    raise ValueError("修改区参数时必须给出 region_index")
                if not (0 <= region_index < len(new_regions)):
                    raise NotFound(
                        f"region_index={region_index} 越界（共 {len(new_regions)} 区）")
                before = copy.deepcopy(new_regions[region_index])
                for key in ("thickness", "D", "sigma_a", "nu_sigma_f", "n_mesh"):
                    val = region_patch.get(key)
                    if val is not None:
                        new_regions[region_index][key] = val
                if new_regions[region_index] != before:
                    geometry_changed = True

            left = patch.get("left_boundary") or cur["left_boundary"]
            right = patch.get("right_boundary") or cur["right_boundary"]
            if left != cur["left_boundary"] or right != cur["right_boundary"]:
                geometry_changed = True

            settings_patch = patch.get("settings")
            new_settings = resolve_settings(
                settings_patch if settings_patch else None)
            # 注意：补丁里没给的设置项应沿用当前版本，而不是落回默认值
            if settings_patch:
                merged = dict(cur["settings"])
                for key in ("k_tol", "flux_tol", "max_iter"):
                    if settings_patch.get(key) is not None:
                        merged[key] = settings_patch[key]
                new_settings = merged
                if new_settings != cur["settings"]:
                    geometry_changed = True

            target = patch.get("target_total_fission")
            if target is None:
                target = cur["target_total_fission"]
            elif target != cur["target_total_fission"]:
                geometry_changed = True

            if name:
                case["name"] = name

            if geometry_changed:
                now = utc_now()
                new_version = {
                    "version": cur["version"] + 1,
                    "created_at": now,
                    "change_summary": _describe_change(cur, patch, region_index),
                    "regions": new_regions,
                    "left_boundary": left,
                    "right_boundary": right,
                    "settings": new_settings,
                    "target_total_fission": target,
                    "result_ids": [],
                }
                case["versions"].append(new_version)
                case["current_version"] = new_version["version"]
                case["updated_at"] = now
            else:
                case["updated_at"] = utc_now()

            self._save_case(case)
            return copy.deepcopy(case)

    def delete_case(self, case_id: str) -> None:
        with self._lock:
            case = self._cases.pop(case_id, None)
            if case is None:
                raise NotFound(f"工况 {case_id} 不存在")
            for v in case["versions"]:
                for rid in v["result_ids"]:
                    try:
                        self._result_path(rid).unlink()
                    except FileNotFoundError:
                        pass
            try:
                self._case_path(case_id).unlink()
            except FileNotFoundError:
                pass

    # ---------- 结果 ----------

    def add_result(self, case_id: str, version: int, result: dict) -> dict:
        with self._lock:
            case = self._cases.get(case_id)
            if case is None:
                raise NotFound(f"工况 {case_id} 不存在")
            ver = next((v for v in case["versions"] if v["version"] == version),
                       None)
            if ver is None:
                raise NotFound(f"工况 {case_id} 没有版本 {version}")
            now = utc_now()
            result_id = new_id("res")
            doc = {
                "id": result_id,
                "case_id": case_id,
                "case_name": case["name"],
                "version": version,
                "created_at": now,
                **result,
            }
            self._atomic_write_json(self._result_path(result_id), doc)
            ver["result_ids"].append(result_id)
            self._save_case(case)
            return copy.deepcopy(doc)

    def get_result(self, result_id: str) -> dict:
        path = self._result_path(result_id)
        if not path.exists():
            raise NotFound(f"结果 {result_id} 不存在")
        with self._lock:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

    def list_results(self, case_id: str, version: Optional[int] = None) -> list[dict]:
        case = self.get_case(case_id)
        docs = []
        versions = [v for v in case["versions"]
                    if version is None or v["version"] == version]
        for v in versions:
            for rid in v["result_ids"]:
                docs.append(self.get_result(rid))
        docs.sort(key=lambda d: d["created_at"])
        return docs

    def latest_converged_result(self, case_id: str,
                                version: int) -> Optional[dict]:
        """取指定版本上最近一次成功结果（热启动用）。"""
        case = self.get_case(case_id)
        ver = next((v for v in case["versions"] if v["version"] == version), None)
        if ver is None or not ver["result_ids"]:
            return None
        # 结果按时间追加，列表最后一个即该版本最新结果
        return self.get_result(ver["result_ids"][-1])

    # ---------- 作业 ----------

    def save_job(self, job: dict) -> None:
        with self._lock:
            self._jobs[job["id"]] = copy.deepcopy(job)
            self._save_job(job)

    def update_job(self, job_id: str, **fields) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise NotFound(f"作业 {job_id} 不存在")
            job.update(fields)
            self._save_job(job)
            return copy.deepcopy(job)

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                path = self._job_path(job_id)
                if path.exists():
                    with open(path, encoding="utf-8") as fh:
                        job = json.load(fh)
                    self._jobs[job_id] = job
                else:
                    raise NotFound(f"作业 {job_id} 不存在")
            return copy.deepcopy(job)

    def list_jobs(self, case_id: Optional[str] = None) -> list[dict]:
        with self._lock:
            jobs = [copy.deepcopy(j) for j in self._jobs.values()]
        if case_id:
            jobs = [j for j in jobs if j["case_id"] == case_id]
        jobs.sort(key=lambda j: j["submitted_at"])
        return jobs

    def recover_interrupted_jobs(self) -> list[str]:
        """启动时把残留在 queued/running 的作业标记为 interrupted（不自动重跑）。"""
        recovered = []
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in ("queued", "running"):
                    job["status"] = "interrupted"
                    job["cancelled"] = True
                    job["finished_at"] = utc_now()
                    job["message"] = "服务重启，作业未完成（不自动恢复）"
                    self._save_job(job)
                    recovered.append(job["id"])
        return recovered


def _describe_change(cur: dict, patch: dict, region_index: Optional[int]) -> str:
    parts = []
    if patch.get("region"):
        keys = [k for k, v in patch["region"].items() if v is not None]
        idx = "?" if region_index is None else region_index
        parts.append(f"修改第 {idx} 区: {', '.join(keys)}")
    if patch.get("left_boundary"):
        parts.append(f"左边界 -> {patch['left_boundary']}")
    if patch.get("right_boundary"):
        parts.append(f"右边界 -> {patch['right_boundary']}")
    if patch.get("settings"):
        parts.append("求解设置变更")
    if patch.get("target_total_fission") is not None:
        parts.append("目标总裂变率变更")
    return "; ".join(parts) or "参数变更"
