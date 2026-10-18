"""Bounded readers for actual trainer JSONL, output images and stdout logs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

from .config import InputError, read_json, safe_path
from .jobs import job_directory

MAX_LOG_CHUNK = 128 * 1024
MAX_METRIC_BYTES = 4 * 1024 * 1024
MAX_METRIC_ROWS = 5000
MAX_IMAGES = 500
MAX_SCANNED_FILES = 10000
PNG_NAMES = re.compile(r"^(?:preview[0-9]+(?:_[0-9]+)?|prompt[0-9]+_seed-?[0-9]+_(?:base|adapter[0-9]+_.+)|base|lora)\.png$", re.IGNORECASE)


def logs(identifier: str, offset: int) -> dict:
    path = safe_path(job_directory(identifier) / "log.txt")
    if not path.exists():
        return {"text": "", "offset": 0}
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        cursor = min(offset, size)
        handle.seek(cursor)
        chunk = handle.read(MAX_LOG_CHUNK)
        # Do not split a trailing UTF-8 code point across polling responses.
        if cursor + len(chunk) < size:
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError as exc:
                if exc.end == len(chunk) and exc.reason == "unexpected end of data":
                    chunk = chunk[:exc.start]
        return {"text": chunk.decode("utf-8", errors="replace"), "offset": cursor + len(chunk)}


def jsonl_tail(path: Path) -> tuple[list[dict], bool]:
    if not path.is_file():
        return [], False
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        start = max(0, size - MAX_METRIC_BYTES)
        handle.seek(start)
        if start:
            handle.readline()
        lines = handle.read(MAX_METRIC_BYTES).splitlines(keepends=True)
    result = []
    for line in lines:
        if not line.endswith(b"\n") or len(line) > 256 * 1024:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                result.append(record)
        except (ValueError, UnicodeError):
            continue
    return result, bool(start)


def metric_rows(job: dict) -> dict:
    path = safe_path(Path(job["output_dir"]) / "train.jsonl")
    records, truncated = jsonl_tail(path)
    series = []
    columns = set()
    for record in records:
        if record.get("event") != "step" or not isinstance(record.get("step"), (int, float)):
            continue
        row = {key: value for key, value in record.items() if type(value) in (int, float) and math.isfinite(value)}
        if "step_seconds" in row:
            row["seconds_per_step"] = row["step_seconds"]
        for optimizer, values in record.get("lr", {}).items() if isinstance(record.get("lr"), dict) else []:
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                if type(value) in (int, float) and math.isfinite(value):
                    row[f"lr.{optimizer}.{index}"] = value
                    row.setdefault("lr", value)
        series.append(row)
        columns.update(row)
    truncated = truncated or len(series) > MAX_METRIC_ROWS
    return {"series": series[-MAX_METRIC_ROWS:], "columns": sorted(columns), "source": str(path) if path.is_file() else None, "truncated": truncated}


def metadata_index(job: dict) -> dict[str, dict]:
    root = safe_path(job["output_dir"])
    metadata = {}
    parameters = safe_path(root / "parameters.json")
    if parameters.is_file():
        try:
            document = read_json(parameters)
            for item in document.get("images", []):
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    continue
                candidate = safe_path(item["path"], root=root)
                data = {**document.get("sample", {}), **item}
                prompts = document.get("prompts", [])
                index = item.get("prompt_index")
                if type(index) is int and 0 <= index < len(prompts):
                    data["prompt"] = prompts[index]
                data["negative_prompt"] = document.get("negative_prompt")
                metadata[str(candidate)] = data
        except (OSError, InputError, TypeError, AttributeError):
            pass  # Partial parameter files are expected while a sample job finishes.
    records, _ = jsonl_tail(safe_path(root / "train.jsonl"))
    for record in records:
        if record.get("event") != "preview" or not isinstance(record.get("preview"), dict):
            continue
        preview = record["preview"]
        for item in preview.get("outputs", [preview]):
            if not isinstance(item, dict) or not isinstance(item.get("output_path"), str):
                continue
            try:
                candidate = safe_path(item["output_path"], root=root)
            except InputError:
                continue
            metadata[str(candidate)] = {**item, "step": record.get("step"), "epoch": record.get("epoch")}
    return metadata


def image_paths(job: dict) -> list[Path]:
    root = safe_path(job["output_dir"])
    if job["stage"] == "cache" or not root.is_dir():
        return []
    candidates = []
    scanned = 0
    for directory, directories, filenames in os.walk(root, followlinks=False):
        allowed = []
        for name in directories:
            scanned += 1
            if scanned > MAX_SCANNED_FILES:
                return sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:MAX_IMAGES]
            try:
                safe_path(Path(directory) / name, root=root)
                allowed.append(name)
            except InputError:
                continue
        directories[:] = allowed
        for name in filenames:
            scanned += 1
            if scanned > MAX_SCANNED_FILES:
                return sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:MAX_IMAGES]
            if not PNG_NAMES.fullmatch(name):
                continue
            try:
                path = safe_path(Path(directory) / name, root=root)
                if path.is_file():
                    candidates.append(path)
            except (OSError, InputError):
                continue
    return sorted(candidates, key=lambda item: item.stat().st_mtime_ns, reverse=True)[:MAX_IMAGES]


def image_id(job: dict, path: Path) -> str:
    return hashlib.sha256(path.relative_to(Path(job["output_dir"])).as_posix().encode("utf-8")).hexdigest()[:32]


def gallery(job: dict) -> dict:
    metadata = metadata_index(job)
    result = []
    for path in image_paths(job):
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            width, height = struct.unpack(">II", header[16:24])
        except OSError:
            continue
        identifier = image_id(job, path)
        result.append({"id": identifier, "name": path.name, "url": f"/api/jobs/{job['id']}/images/{identifier}", "width": width, "height": height, "metadata": metadata.get(str(path), {})})
    return {"images": result}


def resolve_image(job: dict, identifier: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", identifier):
        raise KeyError("图片不存在。")
    for path in image_paths(job):
        if image_id(job, path) == identifier:
            with path.open("rb") as handle:
                if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                    break
            return safe_path(path, root=Path(job["output_dir"]))
    raise KeyError("图片不存在或不属于本任务输出。")
