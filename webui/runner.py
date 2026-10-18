"""Detached supervision; model imports occur only in the --execute child."""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from .config import WORKSPACE_DIR, InputError, safe_path, utc_now
from .jobs import (
    TERMINAL, append_event, file_lock, identified_process, identity,
    job_directory, owned_processes, python_path, read_state,
)


def stop_tree(directory: Path, *, include_supervisor: bool, finalize: bool = True) -> bool:
    """Freeze owned roots, enumerate their descendants, then terminate and escalate."""
    with file_lock(directory / "stop.lock"):
        state = read_state(directory)
        processes = {process.pid: process for process in owned_processes(state, include_supervisor=include_supervisor) if process.pid != os.getpid()}
        records = {}
        for process in list(processes.values()):
            try:
                records[process.pid] = identity(process)
                process.suspend()
            except psutil.Error:
                pass
        # Suspending roots closes the usual spawn-between-enumeration-and-kill race.
        for process in list(processes.values()):
            try:
                for child in process.children(recursive=True):
                    if child.pid == os.getpid():
                        continue
                    record = identity(child)
                    child.suspend()
                    records[child.pid] = record
                    processes[child.pid] = child
            except psutil.Error:
                pass
        waiting = []
        for record in reversed(list(records.values())):
            process = identified_process(record)
            if process is None:
                continue
            try:
                process.terminate()
                if os.name != "nt":
                    process.resume()  # Deliver SIGTERM to a stopped Unix process.
                waiting.append(process)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(waiting, timeout=2)
        for process in alive:
            if identified_process(records.get(process.pid)) is not None:
                try:
                    process.kill()
                except psutil.Error:
                    pass
        psutil.wait_procs(alive, timeout=3)
        survivors = [process for record in records.values() if (process := identified_process(record)) is not None]
        for process in survivors:
            try:
                process.resume()
            except psutil.Error:
                pass
        append_event(directory, "tree_stop", descendants=[record for pid, record in records.items() if pid != state.get("supervisor_pid")])
        if survivors:
            # Do not report stopped while the owned process tree still exists.
            append_event(directory, "stop_incomplete", status="stopping", error="部分任务进程未能终止，请再次点击停止。PIDs: " + ", ".join(str(process.pid) for process in survivors))
            return False
        if finalize:
            state = read_state(directory)
            child_process = processes.get(state.get("pid"))
            exit_code = getattr(child_process, "returncode", None)
            if exit_code is None:
                exit_code = state.get("child_exit_code")
            append_event(directory, "stopped", status="stopped", finished_at=utc_now(), exit_code=exit_code, error=None)
        return True


def execute_child(directory: Path) -> int:
    state = read_state(directory)
    process = psutil.Process()
    append_event(directory, "child_started", child=identity(process), pid=process.pid, started_at=utc_now(), status="running")
    if safe_path(directory / "stop.request").exists():
        append_event(directory, "child_cancelled", child_exit_code=0)
        return 0
    command = state["command"]
    if command[:4] not in (
        [str(python_path()), "-u", "-m", "moss_trainer.launch"],
        [str(python_path()), "-u", "-m", "moss_trainer.samples"],
    ):
        raise InputError("拒绝执行非训练器命令。")
    # runpy preserves launch.py's __main__ guard and Windows DataLoader spawn rules.
    # The process records its own identity before importing any heavy model modules.
    sys.argv = [command[3], *command[4:]]
    exit_code = 0
    error = None
    try:
        from ..launch import CACHE_ENVIRONMENT, TEMP

        output = safe_path(state["output_dir"])
        if output.exists():
            raise InputError(f"输出目录已存在，拒绝覆盖: {output}")
        safe_path(state["config_path"])
        for directory_name in CACHE_ENVIRONMENT.values():
            safe_path(TEMP / directory_name)
        runpy.run_module(command[3], run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if exit_code:
            error = f"训练器退出码 {exit_code}" if isinstance(exc.code, (int, type(None))) else str(exc.code)
            if not isinstance(exc.code, (int, type(None))):
                print(exc.code, file=sys.stderr, flush=True)
    except BaseException as exc:
        traceback.print_exc()
        exit_code = 1
        error = f"{type(exc).__name__}: {exc}"
    finally:
        append_event(directory, "child_exited", child_exit_code=exit_code, exit_code=exit_code, child_error=error)
        # If the supervisor crashed, the child still persists a truthful terminal result.
        latest = read_state(directory)
        if identified_process(latest.get("supervisor")) is None:
            stopped = safe_path(directory / "stop.request").exists()
            append_event(directory, "orphan_finished", status="stopped" if stopped else ("completed" if exit_code == 0 else "failed"), finished_at=utc_now(), error=error)
    return exit_code


def supervise(directory: Path) -> int:
    with file_lock(directory / "runner.lock", blocking=False):
        current = psutil.Process()
        append_event(directory, "supervisor_started", supervisor=identity(current), supervisor_pid=current.pid)
        if safe_path(directory / "stop.request").exists():
            append_event(directory, "cancelled_before_start", status="stopped", finished_at=utc_now())
            return 0
        if read_state(directory)["status"] in TERMINAL:
            return 0
        command = [str(python_path()), "-u", "-m", "moss_trainer.webui.runner", str(directory), "--execute"]
        with safe_path(directory / "log.txt").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=WORKSPACE_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        try:
            record = identity(psutil.Process(child.pid))
            append_event(directory, "child_spawned", child=record, pid=child.pid)
        except psutil.NoSuchProcess:
            pass
        known = set()
        while child.poll() is None:
            state = read_state(directory)
            if safe_path(directory / "stop.request").exists():
                if stop_tree(directory, include_supervisor=False):
                    child.wait()
                return 0
            descendants = []
            for process in owned_processes(state):
                try:
                    descendants.append(identity(process))
                except psutil.Error:
                    continue
            keys = {(record["pid"], record["create_time"]) for record in descendants}
            if keys != known:
                append_event(directory, "process_tree", descendants=descendants)
                known = keys
            time.sleep(0.25)
        exit_code = child.returncode
        state = read_state(directory)
        stopped = safe_path(directory / "stop.request").exists()
        # A failed interpreter may leave DataLoader/compiler descendants alive.
        if owned_processes(state) and not stop_tree(directory, include_supervisor=False, finalize=False):
            return 1
        error = state.get("child_error")
        if exit_code and not error and not stopped:
            error = f"训练进程退出码 {exit_code}；请查看日志。"
        append_event(directory, "finished", status="stopped" if stopped else ("completed" if exit_code == 0 else "failed"), exit_code=exit_code, finished_at=utc_now(), error=error)
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    directory = job_directory(args.directory.name)
    if safe_path(args.directory) != directory:
        parser.error("Job directory does not match the recorded workspace state.")
    try:
        if args.execute:
            code = execute_child(directory)
        elif args.stop:
            stop_tree(directory, include_supervisor=True)
            code = 0
        else:
            code = supervise(directory)
    except BaseException as exc:
        traceback.print_exc()
        if not isinstance(exc, SystemExit):
            append_event(directory, "runner_error", error=f"监督进程错误: {type(exc).__name__}: {exc}")
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
