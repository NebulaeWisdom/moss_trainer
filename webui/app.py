"""Loopback-only HTTP control surface; no model imports in this process."""
from __future__ import annotations

import ipaddress
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import artifacts
from .config import (
    DEFAULT_OUTPUT_ROOT, MAX_JSON_BYTES, PROJECT_DIR, STATE_DIR, WORKSPACE_DIR,
    ConfigStore, InputError, evaluation_defaults, read_json, safe_path, schema,
)
from .jobs import JobStore, checkpoint_config, public_job, python_path
from .telemetry import SystemMonitor


def local_hostname(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False


class LocalOnly:
    """Host + same-origin + Fetch Metadata protection against localhost CSRF/rebinding."""
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {}
        for key, value in scope.get("headers", []):
            name = key.decode("latin-1").lower()
            if name in headers and name in {"host", "origin", "content-length"}:
                await JSONResponse({"detail": "重复的安全请求头。"}, status_code=400)(scope, receive, send)
                return
            headers[name] = value.decode("latin-1")
        try:
            host = urlsplit("http://" + headers.get("host", ""))
            port = host.port or 80
            valid = local_hostname(host.hostname) and not host.username and not host.password and not host.path and not host.query and not host.fragment
            server = scope.get("server")
            valid = valid and (not server or port == server[1])
            origin_value = headers.get("origin")
            if origin_value is not None:
                origin = urlsplit(origin_value)
                valid = valid and origin.scheme == scope.get("scheme", "http") and origin.hostname == host.hostname and (origin.port or (443 if origin.scheme == "https" else 80)) == port and not origin.username and not origin.password and origin.path in ("", "/") and not origin.query and not origin.fragment
            if scope["path"].startswith("/api/") and headers.get("sec-fetch-site") == "cross-site":
                valid = False
        except ValueError:
            valid = False
        if not valid:
            await JSONResponse({"detail": "仅允许本机同源请求；请从 WebUI 本地地址打开页面。"}, status_code=403)(scope, receive, send)
            return
        if scope["method"] in {"POST", "PUT", "PATCH"} and scope["path"].startswith("/api/"):
            try:
                length = int(headers.get("content-length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_JSON_BYTES:
                await JSONResponse({"detail": "请求体超过 2 MiB 上限或长度无效。"}, status_code=413)(scope, receive, send)
                return
            if length and headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
                await JSONResponse({"detail": "控制 API 仅接受 application/json。"}, status_code=415)(scope, receive, send)
                return
        received = 0

        async def bounded_receive() -> dict:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_JSON_BYTES:
                raise HTTPException(status_code=413, detail="请求体超过 2 MiB 上限。")
            return message

        async def secure_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                additions = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"), (b"referrer-policy", b"same-origin"), (b"content-security-policy", b"frame-ancestors 'none'")]
                if scope["path"].startswith("/api/"):
                    additions.append((b"cache-control", b"no-store"))
                message = {**message, "headers": [*message.get("headers", []), *additions]}
            await send(message)

        await self.app(scope, bounded_receive, secure_send)


class RevisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="config", max_length=128)
    config: dict[str, Any]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stage: Literal["cache", "train", "evaluation"]
    config: dict[str, Any]
    name: str | None = Field(default=None, max_length=128)
    cache_dir: str | None = None
    output_dir: str | None = None
    adapters: list[str] = Field(default_factory=list)
    include_base: bool = False
    resume: str | None = None


def create_app() -> FastAPI:
    # Windows registry associations must not turn ES modules into text/plain.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/javascript", ".mjs")
    configs = ConfigStore()
    jobs = JobStore()
    monitor = SystemMonitor(WORKSPACE_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        monitor.close()
        # Deliberately do not stop jobs when the HTTP process shuts down.

    app = FastAPI(title="MOSS Local Workbench", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(LocalOnly)

    @app.exception_handler(InputError)
    async def input_error(request: Request, exc: InputError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(KeyError)
    async def missing_error(request: Request, exc: KeyError):
        return JSONResponse({"detail": str(exc.args[0]) if exc.args else "记录不存在。"}, status_code=404)

    @app.exception_handler(OSError)
    async def filesystem_error(request: Request, exc: OSError):
        return JSONResponse({"detail": f"本地文件或进程操作失败: {exc}"}, status_code=422)

    @app.get("/api/info")
    def info():
        return {"project_dir": str(PROJECT_DIR), "workspace_dir": str(WORKSPACE_DIR), "python": str(python_path()), "state_dir": str(STATE_DIR), "default_output_root": str(DEFAULT_OUTPUT_ROOT), "stages": ["cache", "train"], "evaluation_stage": "evaluation"}

    @app.get("/api/schema")
    def config_schema():
        return schema()

    @app.get("/api/evaluation-defaults")
    def get_evaluation_defaults():
        return {"config": evaluation_defaults()}

    @app.get("/api/configs")
    def list_configs():
        entries = configs.entries()
        default = next((item["id"] for item in entries if item["id"] == "builtin:moss_official"), entries[0]["id"] if entries else None)
        return {"configs": entries, "default_id": default}

    @app.get("/api/config")
    def get_config(id: str = Query(..., max_length=256)):
        return configs.get(id)

    @app.post("/api/configs", status_code=201)
    def save_config(request: RevisionRequest):
        return configs.save(request.name, request.config)

    @app.get("/api/checkpoints")
    def list_checkpoints():
        return jobs.checkpoints()

    @app.get("/api/checkpoints/config")
    def get_checkpoint_config(path: str = Query(...)):
        return checkpoint_config(path)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [public_job(state) for state in jobs.states()]}

    @app.post("/api/jobs", status_code=201)
    def start_job(request: JobRequest):
        return jobs.create(request.model_dump())

    @app.get("/api/jobs/{identifier}")
    def get_job(identifier: str):
        return public_job(jobs.state(identifier))

    @app.get("/api/jobs/{identifier}/config")
    def get_job_config(identifier: str):
        state = jobs.state(identifier)
        path = safe_path(state["config_path"])
        return {"config": read_json(path), "path": str(path)}

    @app.post("/api/jobs/{identifier}/stop")
    def stop_job(identifier: str):
        return jobs.stop(identifier)

    @app.get("/api/jobs/{identifier}/logs")
    def get_logs(identifier: str, offset: int = Query(default=0, ge=0)):
        return artifacts.logs(identifier, offset)

    @app.get("/api/jobs/{identifier}/metrics")
    def get_metrics(identifier: str):
        return artifacts.metric_rows(jobs.state(identifier))

    @app.get("/api/jobs/{identifier}/images")
    def get_images(identifier: str):
        return artifacts.gallery(jobs.state(identifier))

    @app.get("/api/jobs/{identifier}/images/{image_identifier}")
    def get_image(identifier: str, image_identifier: str):
        path = artifacts.resolve_image(jobs.state(identifier), image_identifier)
        return FileResponse(path, media_type="image/png", filename=path.name, content_disposition_type="inline")

    @app.get("/api/system")
    def get_system():
        return monitor.sample()

    static = safe_path(Path(__file__).parent / "static")
    release = None
    if static.is_dir():
        candidates = []
        for path in static.iterdir():
            try:
                candidate = safe_path(path)
                if candidate.is_dir() and safe_path(candidate / "index.html").is_file():
                    candidates.append(candidate)
            except InputError:
                continue
        if candidates:
            release = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        elif safe_path(static / "index.html").is_file():
            release = static
    if release is not None:
        app.mount("/", StaticFiles(directory=release, html=True, follow_symlink=False), name="frontend")
    else:
        @app.get("/")
        def missing_frontend():
            return JSONResponse({"detail": "前端构建产物不存在。请在 moss_trainer/webui/frontend 中执行 npm install 与 npm run build 后重启 WebUI。"}, status_code=503)
    return app
