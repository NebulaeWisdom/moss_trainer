"""Disk-backed jobs, append-only lifecycle journal and process identities."""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import psutil

from .config import (
    DEFAULT_OUTPUT_ROOT, PROJECT_DIR, STATE_DIR, WORKSPACE_DIR, InputError,
    input_path, new_id, read_json, safe_path, utc_now, validate_config, write_new_json,
)

TERMINAL = {"completed", "failed", "stopped", "interrupted"}
JOB_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{32}$")
_THREAD_LOCK = threading.RLock()
MAX_CHECKPOINT_ENTRIES = 10000
MAX_CHECKPOINTS = 500
MAX_CHECKPOINT_HEADER = 4 * 1024 * 1024
CHECKPOINT_NAME = re.compile(r"^(?:adapter|step([0-9]+))\.safetensors$", re.IGNORECASE)


def checkpoint_path(value: str) -> Path:
    if not value.strip():
        raise InputError("检查点路径不能为空。")
    path = safe_path(value)
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise InputError(f"检查点必须是工作区内实际存在的 .safetensors 文件: {path}")
    return path


def checkpoint_metadata(path: Path) -> dict:
    """Read only the bounded safetensors JSON header, never tensors or pickle."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            size = int.from_bytes(prefix, "little")
            if len(prefix) != 8 or not 2 <= size <= MAX_CHECKPOINT_HEADER:
                raise InputError(f"检查点头无效或超过 4 MiB 上限: {path.name}")
            payload = handle.read(size)
            if len(payload) != size:
                raise InputError(f"检查点尚未写完或文件损坏: {path.name}")
            header = json.loads(payload)
            if not isinstance(header, dict):
                raise ValueError("header is not an object")
            data_bytes = os.fstat(handle.fileno()).st_size - 8 - size
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if not tensors or any(not isinstance(value, dict) or not isinstance(value.get("data_offsets"), list) or len(value["data_offsets"]) != 2 or any(type(offset) is not int for offset in value["data_offsets"]) or not 0 <= value["data_offsets"][0] <= value["data_offsets"][1] <= data_bytes for value in tensors):
                raise ValueError("tensor data is incomplete")
            metadata = header.get("__metadata__", {})
            return metadata if isinstance(metadata, dict) else {}
    except (UnicodeError, ValueError) as exc:
        raise InputError(f"检查点头无效或文件尚未写完: {path.name}: {exc}") from exc


def checkpoint_config(value: str) -> dict:
    checkpoint = checkpoint_path(value)
    checkpoint_metadata(checkpoint)
    path = safe_path(checkpoint.parent / "config.json")
    if not path.is_file():
        raise InputError(f"检查点同目录没有 config.json；请显式配置匹配的评估模型与适配器参数: {checkpoint.parent}")
    return {"config": validate_config(read_json(path)), "path": str(path)}


def validate_checkpoint_config(path: Path, config: dict) -> None:
    metadata = checkpoint_metadata(path)
    adapter = config["adapter"]
    conflicts = []
    for key, expected in (("ss_arch", "MOSSImageTransformer"), ("ss_model_type", "MOSSImage"), ("ss_network_module", f"lycoris.{adapter['algo']}")):
        if key in metadata and metadata[key] != expected:
            conflicts.append(key)
    for key, expected in (("ss_network_dim", adapter["rank"]), ("ss_network_alpha", adapter["alpha"])):
        if key in metadata:
            try:
                matches = float(metadata[key]) == float(expected)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                conflicts.append(key)
    if isinstance(metadata.get("ss_network_args"), str):
        try:
            arguments = json.loads(metadata["ss_network_args"])
        except ValueError as exc:
            raise InputError(f"检查点适配器元数据不是有效 JSON: {path.name}") from exc
        if isinstance(arguments, dict):
            expected_arguments = {"algo": adapter["algo"], "target_modules": adapter["target_modules"], **adapter.get("kwargs", {})}
            for key, value in arguments.items():
                if key != "dropout" and (key not in expected_arguments or expected_arguments[key] != value):
                    conflicts.append("ss_network_args." + key)
    sibling = safe_path(path.parent / "config.json")
    if sibling.is_file():
        source = read_json(sibling)
        source_adapter = source.get("adapter", {}) if isinstance(source, dict) else {}
        if not isinstance(source_adapter, dict):
            raise InputError(f"检查点同目录 config.json 的 adapter 必须为对象: {path.parent}")
        for key in ("algo", "rank", "alpha", "target_modules", "kwargs"):
            if source_adapter.get(key, {} if key == "kwargs" else None) != adapter.get(key, {} if key == "kwargs" else None):
                conflicts.append("adapter." + key)
    if conflicts:
        raise InputError(f"检查点 {path.name} 与当前评估配置不匹配 ({', '.join(conflicts)})。所有来源共享一套适配器结构；请加载匹配配置或分开评估，不能跳过不兼容来源。")


def python_path() -> Path:
    path = WORKSPACE_DIR / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_file():
        raise InputError(f"项目 venv Python 不存在: {path}")
    return path


def job_directory(identifier: str) -> Path:
    if not JOB_ID.fullmatch(identifier):
        raise KeyError("任务不存在。")
    path = safe_path(STATE_DIR / "jobs" / identifier)
    if not path.is_dir():
        raise KeyError("任务不存在。")
    return path


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Persistent lock file: never unlink locks or replace persisted job data."""
    path = safe_path(path)
    with _THREAD_LOCK, path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            deadline = time.monotonic() + (10 if blocking else 0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise InputError("任务状态正在被另一进程更新，请稍后重试。")
                    time.sleep(0.025)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_state(directory: Path) -> dict:
    base = read_json(safe_path(directory / "job.json"))
    journal = safe_path(directory / "events.jsonl")
    if not journal.exists():
        return base
    with journal.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        start = max(0, size - 1024 * 1024)
        handle.seek(start)
        if start:
            handle.readline()
        lines = handle.read(1024 * 1024).splitlines(keepends=True)
    for line in reversed(lines):
        if not line.endswith(b"\n"):
            continue
        try:
            event = json.loads(line)
            if isinstance(event.get("state"), dict):
                return {**base, **event["state"]}
        except (ValueError, UnicodeError):
            continue
    return base


def append_event(directory: Path, event: str, **updates: Any) -> dict:
    with file_lock(directory / "events.lock"):
        state = read_state(directory)
        # A late reconciliation must not overwrite the child's actual exit result.
        if state["status"] in TERMINAL and updates.get("status") in {"queued", "running", "stopping", "interrupted"}:
            updates.pop("status", None)
            updates.pop("error", None)
            updates.pop("finished_at", None)
        state.update(updates)
        record = {"event": event, "at": utc_now(), "state": state}
        payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with safe_path(directory / "events.jsonl").open("ab", buffering=0) as handle:
            handle.write(payload)
            os.fsync(handle.fileno())
        return state


def identity(process: psutil.Process) -> dict:
    return {"pid": process.pid, "create_time": process.create_time()}


def identified_process(record: dict | None) -> psutil.Process | None:
    if not record:
        return None
    try:
        process = psutil.Process(int(record["pid"]))
        if abs(process.create_time() - float(record["create_time"])) < 0.001 and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except (psutil.Error, KeyError, ValueError, TypeError):
        pass
    return None


def owned_processes(state: dict, *, include_supervisor: bool = False) -> list[psutil.Process]:
    records = [state.get("child"), *state.get("descendants", [])]
    if include_supervisor:
        records.append(state.get("supervisor"))
    result = {}
    for record in records:
        process = identified_process(record)
        if process is not None:
            result[process.pid] = process
    # Descendants are discovered only from identity-checked owned roots.
    for process in list(result.values()):
        try:
            for child in process.children(recursive=True):
                child.create_time()
                result[child.pid] = child
        except psutil.Error:
            pass
    return list(result.values())


def public_job(state: dict) -> dict:
    result = {key: value for key, value in state.items() if key not in {"supervisor", "child", "descendants", "launch_time", "resume_source"}}
    if result["stage"] == "sample":
        # Read compatibility only: immutable historical job files remain unchanged.
        mode = result.pop("sample_mode", "adapter")
        adapter = result.pop("adapter", None)
        result.update(stage="evaluation", adapters=[adapter] if adapter and mode != "base" else [], include_base=mode in {"base", "compare"})
    return result


def detached_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def launch_supervisor(directory: Path, *, stopping: bool = False) -> dict:
    command = [str(python_path()), "-u", "-m", "moss_trainer.webui.runner", str(directory)]
    if stopping:
        command.append("--stop")
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    # A file descriptor inherited by the runner is independent of HTTP/UI lifetime.
    with safe_path(directory / "log.txt").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=WORKSPACE_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=environment, close_fds=True, **detached_options())
    try:
        record = identity(psutil.Process(process.pid))
    except psutil.NoSuchProcess:
        return read_state(directory)
    return append_event(directory, "stopper_spawned" if stopping else "supervisor_spawned", **({} if stopping else {"supervisor": record, "supervisor_pid": process.pid}))


class JobStore:
    def __init__(self) -> None:
        safe_path(STATE_DIR / "jobs").mkdir(parents=True, exist_ok=True)

    def states(self) -> list[dict]:
        result = []
        for path in sorted(safe_path(STATE_DIR / "jobs").iterdir(), reverse=True):
            if not JOB_ID.fullmatch(path.name):
                continue
            try:
                result.append(self.state(path.name))
            except (KeyError, OSError, InputError):
                continue
        return result

    def state(self, identifier: str) -> dict:
        directory = job_directory(identifier)
        state = read_state(directory)
        children = owned_processes(state)
        if state["status"] in TERMINAL:
            if children:
                return {**state, "status": "stopping" if safe_path(directory / "stop.request").exists() else "running", "error": "训练主体已结束，但仍有已记录的子进程；可显式停止整个进程树。"}
            return state
        supervisor = identified_process(state.get("supervisor"))
        if supervisor is None and not children and time.time() - state["launch_time"] > 15:
            stopping = safe_path(directory / "stop.request").exists()
            state = append_event(directory, "reconciled", status="stopped" if stopping else "interrupted", finished_at=utc_now(), error="停止请求已执行。" if stopping else "监督进程与训练进程均已退出，未留下完成事件。请查看日志。")
        elif supervisor is None and children:
            state = {**state, "status": "stopping" if safe_path(directory / "stop.request").exists() else "running", "error": "监督进程已退出；已记录的训练子进程仍在运行，可显式停止。"}
        return state

    def checkpoints(self) -> dict:
        project_runs = safe_path(PROJECT_DIR / "runs")
        names = {}
        roots = [project_runs] if project_runs.is_dir() else []
        for state in self.states():
            if state["stage"] != "train":
                continue
            try:
                root = safe_path(state["output_dir"])
            except InputError:
                continue
            names[root] = state["name"]
            if root.is_dir() and not any(root.is_relative_to(existing) for existing in roots):
                roots.append(root)
        pending = list(reversed(roots))
        seen = set()
        result = []
        scanned = 0
        header_bytes = 0
        truncated = False
        while pending and not truncated:
            directory = pending.pop()
            if directory in seen:
                continue
            seen.add(directory)
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        scanned += 1
                        if scanned > MAX_CHECKPOINT_ENTRIES or len(result) >= MAX_CHECKPOINTS:
                            truncated = True
                            break
                        try:
                            path = safe_path(entry.path)
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(path)
                                continue
                            match = CHECKPOINT_NAME.fullmatch(entry.name)
                            if match is None or not entry.is_file(follow_symlinks=False):
                                continue
                            with path.open("rb") as handle:
                                header_size = int.from_bytes(handle.read(8), "little")
                            header_bytes += min(header_size, MAX_CHECKPOINT_HEADER)
                            if header_bytes > 64 * 1024 * 1024:
                                truncated = True
                                break
                            metadata = checkpoint_metadata(path)
                            if metadata.get("ss_arch") != "MOSSImageTransformer" and metadata.get("ss_model_type") != "MOSSImage":
                                continue
                            item = {"path": str(path), "name": path.name, "run_name": names.get(path.parent, path.parent.name)}
                            step = match.group(1) if match.group(1) is not None else metadata.get("ss_training_steps")
                            if isinstance(step, (str, int)) and str(step).isdigit():
                                item["step"] = int(step)
                            config_path = safe_path(path.parent / "config.json")
                            if config_path.is_file():
                                item["config_path"] = str(config_path)
                            result.append(item)
                        except (InputError, OSError):
                            continue
            except OSError:
                continue
        result.sort(key=lambda item: (item["run_name"], item.get("step", -1), item["name"]), reverse=True)
        return {"checkpoints": result, "truncated": truncated}

    def create(self, request: dict) -> dict:
        from ..launch import CACHE_ENVIRONMENT, TEMP

        for directory_name in CACHE_ENVIRONMENT.values():
            safe_path(TEMP / directory_name)
        stage = request["stage"]
        config = request["config"]
        resume = request.get("resume")
        if resume and stage != "train":
            raise InputError("resume 仅用于训练。")
        if stage != "evaluation" and (request.get("adapters") or request.get("include_base")):
            raise InputError("adapters/include_base 仅用于独立 LoRA 评估；训练适配器设置位于 config.adapter。")
        if stage == "evaluation" and request.get("cache_dir"):
            raise InputError("LoRA 评估不使用训练缓存，请移除 cache_dir。")
        cache = None
        resume_source = None
        if resume:
            resume_path = safe_path(resume)
            if not resume_path.is_file() or resume_path.suffix != ".pt":
                raise InputError("resume 必须是工作区内可信的完整 .pt 状态文件，不是适配器权重。")
            sibling_config = safe_path(resume_path.parent / "config.json")
            if not sibling_config.is_file():
                raise InputError("恢复状态所在目录缺少 config.json；WebUI 需要训练器生成的配置来展示快照。无此文件时请使用原 CLI。")
            config = read_json(sibling_config)
            for previous in self.states():
                if previous["stage"] == "train" and resume_path.parent == Path(previous["output_dir"]):
                    resume_source = previous
                    cache = Path(previous["cache_dir"]) if previous.get("cache_dir") else None
                    break
            if cache is None:
                source_log = safe_path(resume_path.parent / "train.jsonl")
                if source_log.is_file():
                    with source_log.open("rb") as handle:
                        first_line = handle.readline(256 * 1024)
                    try:
                        start = json.loads(first_line)
                        if start.get("event") == "start" and isinstance(start.get("cache_dir"), str):
                            cache = Path(start["cache_dir"])
                    except (ValueError, UnicodeError, AttributeError):
                        pass
        config = validate_config(config, stage)
        if stage == "train" and not resume:
            if not request.get("cache_dir"):
                raise InputError("训练需要 cache_dir：先创建缓存，再填入缓存任务的实际目录。")
            cache = input_path(request["cache_dir"], "cache_dir", directory=True)
        if cache is not None:
            for filename in ("manifest.json", "text_manifest.json", "latent_stats.safetensors"):
                if not (cache / filename).is_file():
                    raise InputError(f"缓存不完整，缺少 {filename}: {cache}")
        adapters = []
        include_base = request.get("include_base", False)
        if stage == "evaluation":
            for value in request.get("adapters", []):
                path = checkpoint_path(value)
                if path in adapters:
                    raise InputError(f"评估来源重复: {path}。请保留一次；输入顺序即矩阵列顺序。")
                adapters.append(path)
            if not include_base and not adapters:
                raise InputError("评估至少选择基模或一个适配器检查点。")
            if adapters:
                missing = [key for key in ("algo", "rank", "alpha", "target_modules", "dropout") if key not in config.get("adapter", {})]
                if missing:
                    raise InputError("评估适配器缺少配置字段: " + ", ".join("adapter." + key for key in missing))
                for path in adapters:
                    validate_checkpoint_config(path, config)
        parent = safe_path(request.get("output_dir") or DEFAULT_OUTPUT_ROOT)
        if parent.exists() and not parent.is_dir():
            raise InputError(f"输出父路径不是目录: {parent}")
        cache_parent = safe_path(request.get("cache_dir") or (WORKSPACE_DIR / "temp" / "webui-cache")) if stage == "cache" else None
        if cache_parent is not None and cache_parent.exists() and not cache_parent.is_dir():
            raise InputError(f"缓存父路径不是目录: {cache_parent}")
        with file_lock(STATE_DIR / "launch.lock"):
            for previous in self.states():
                if previous["status"] not in TERMINAL or owned_processes(previous, include_supervisor=True):
                    raise InputError(f"已有 GPU 任务 {previous['name']} ({previous['id']}) 处于 {previous['status']}。请等待或显式停止后再启动。")
            identifier = new_id()
            directory = safe_path(STATE_DIR / "jobs" / identifier)
            directory.mkdir()
            snapshot = directory / "config.json"
            write_new_json(snapshot, config)
            output = safe_path(parent / identifier)
            if stage == "cache":
                assert cache_parent is not None
                cache = safe_path(cache_parent / identifier)
                output = cache
            if output.exists():
                raise InputError(f"拒绝复用已有输出目录: {output}")
            command = [str(python_path()), "-u", "-m", "moss_trainer.launch", stage]
            if stage == "cache":
                command += ["--config", str(snapshot), "--cache-dir", str(cache)]
            elif stage == "train":
                command += ["--resume", str(resume_path)] if resume else ["--config", str(snapshot), "--cache-dir", str(cache)]
                command += ["--output-dir", str(output)]
            else:
                command = [str(python_path()), "-u", "-m", "moss_trainer.samples", "--config", str(snapshot)]
                if adapters:
                    command += ["--adapter", *(str(path) for path in adapters)]
                multipliers = (["0"] if include_base else []) + (["1"] if adapters else [])
                command += ["--multiplier", *multipliers, "--output-dir", str(output)]
            state = {"id": identifier, "name": request.get("name") or f"{stage}-{identifier}", "stage": stage, "status": "queued", "created_at": utc_now(), "launch_time": time.time(), "config_path": str(snapshot), "output_dir": str(output), "command": command, "cache_dir": str(cache) if cache else None, "pid": None, "exit_code": None}
            if stage == "evaluation":
                state["adapters"] = [str(path) for path in adapters]
                state["include_base"] = include_base
                state["compatibility_note"] = "所有检查点共享此评估配置的基础模型与适配器结构（算法、rank、alpha、目标模块及 kwargs）；已知元数据冲突会拒绝，最终张量兼容性由训练器 strict=True 加载验证，失败不会跳过来源。"
            if resume:
                state["resume"] = str(resume_path)
                state["resume_source"] = resume_source["id"] if resume_source else None
                state["config_origin"] = "恢复任务快照取自同目录 config.json；训练器以可信 .pt 内嵌配置与缓存为准，表单不覆盖恢复状态。"
            write_new_json(directory / "job.json", state)
            append_event(directory, "created")
            try:
                launch_supervisor(directory)
            except (OSError, InputError) as exc:
                append_event(directory, "launch_failed", status="failed", finished_at=utc_now(), error=f"无法启动监督进程: {exc}")
            return public_job(self.state(identifier))

    def stop(self, identifier: str) -> dict:
        directory = job_directory(identifier)
        state = self.state(identifier)
        if state["status"] in TERMINAL and not owned_processes(state, include_supervisor=True):
            return public_job(state)
        try:
            with safe_path(directory / "stop.request").open("x", encoding="utf-8") as handle:
                handle.write(utc_now() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            pass
        state = append_event(directory, "stop_requested", status="stopping")
        # The stop helper outlives the browser/server and also handles an orphaned runner.
        launch_supervisor(directory, stopping=True)
        return public_job(state)
