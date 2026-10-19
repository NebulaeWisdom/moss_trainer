"""Bounded readers for actual trainer JSONL, output images and stdout logs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

from .config import InputError, evaluation_defaults, read_json, safe_path
from .jobs import job_directory, public_job

MAX_LOG_CHUNK = 128 * 1024
MAX_METRIC_BYTES = 4 * 1024 * 1024
MAX_METRIC_ROWS = 5000
MAX_IMAGES = 500
MAX_SCANNED_FILES = 10000
PREVIEW_NAME = re.compile(r"^preview(?P<step>[0-9]+)(?:_(?P<index>[0-9]+))?\.png$", re.IGNORECASE)
EVALUATION_NAME = re.compile(r"^prompt(?P<index>[0-9]+)_seed(?P<seed>-?[0-9]+)_(?:base|adapter(?P<adapter_index>[0-9]+)_.+_m(?P<multiplier>-?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?))\.png$", re.IGNORECASE)
OPTIONAL_SAMPLING_FIELDS = ("scheduler", "uncond_skip_layers", "slg_layers", "slg_scale", "slg_start", "slg_stop")
SAMPLING_FIELDS = ("negative_prompt", "seed", "width", "height", "num_inference_steps", "guidance_scale", *OPTIONAL_SAMPLING_FIELDS)
PROBE_FIELDS = ("prompt_index", "prompt", *SAMPLING_FIELDS)


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
    if job["stage"] != "train":
        return {"series": [], "columns": [], "source": None, "truncated": False}
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
    if job["stage"] in {"evaluation", "sample"}:
        parameters = safe_path(root / "parameters.json")
        if parameters.is_file():
            try:
                document = read_json(parameters)
                declared = document.get("config", {}).get("sample", {})
                defaults = evaluation_defaults()["sample"]
                inference_settings = {key: declared.get(key, defaults[key]) for key in OPTIONAL_SAMPLING_FIELDS}
                if isinstance(inference_settings["scheduler"], dict):
                    inference_settings["scheduler"] = {**defaults["scheduler"], **inference_settings["scheduler"]}
                for item in document.get("images", []):
                    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                        continue
                    candidate = safe_path(item["path"], root=root)
                    data = {**inference_settings, **document.get("sample", {}), **item, "kind": "evaluation", "metadata_source": "parameters.json"}
                    if isinstance(document.get("model_paths"), dict):
                        data["model_paths"] = document["model_paths"]
                    prompts = document.get("prompts", [])
                    index = item.get("prompt_index")
                    if type(index) is int and 0 <= index < len(prompts):
                        data["prompt"] = prompts[index]
                    data["negative_prompt"] = document.get("negative_prompt")
                    metadata[str(candidate)] = data
            except (OSError, InputError, TypeError, AttributeError):
                pass  # The matrix remains incomplete until actual images/metadata arrive.
        return metadata
    if job["stage"] != "train":
        return metadata
    records, _ = jsonl_tail(safe_path(root / "train.jsonl"))
    for record in records:
        if record.get("event") != "preview" or not isinstance(record.get("preview"), dict):
            continue
        preview = record["preview"]
        for index, item in enumerate(preview.get("outputs", [preview])):
            if not isinstance(item, dict) or not isinstance(item.get("output_path"), str):
                continue
            try:
                candidate = safe_path(item["output_path"], root=root)
            except InputError:
                continue
            metadata[str(candidate)] = {**item, "prompt_index": index, "kind": "training_preview", "metadata_source": "train.jsonl", "step": record.get("step"), "epoch": record.get("epoch")}
    return metadata


def sampling_snapshot(job: dict) -> dict:
    try:
        config = read_json(safe_path(job["config_path"]))
        sample = config.get("sample", {})
        if not isinstance(sample, dict):
            return {}
        defaults = evaluation_defaults()["sample"]
        effective = dict(sample)
        for key in OPTIONAL_SAMPLING_FIELDS:
            effective.setdefault(key, defaults[key])
        if isinstance(effective["scheduler"], dict):
            effective["scheduler"] = {**defaults["scheduler"], **effective["scheduler"]}
        return effective
    except (OSError, InputError, AttributeError):
        return {}


def enrich_image(job: dict, path: Path, sample: dict, recorded: dict) -> dict:
    """Attach actual provenance; filenames recover indices without merging equal prompts."""
    data = {key: sample[key] for key in SAMPLING_FIELDS if key in sample}
    prompts = sample.get("prompt")
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(prompts, list):
        prompts = []
    if job["stage"] == "train":
        match = PREVIEW_NAME.fullmatch(path.name)
        if match is not None:
            data.update(step=int(match["step"]), prompt_index=int(match["index"] or 0))
        data.update({key: value for key, value in recorded.items() if value is not None})
        index = data.get("prompt_index")
        if type(index) is int and 0 <= index < len(prompts) and isinstance(prompts[index], str):
            data["prompt"] = prompts[index]
        data["kind"] = "training_preview"
        data.setdefault("metadata_source", "filename+config_snapshot")
        data["probe_complete"] = all(key in data and data[key] is not None for key in PROBE_FIELDS)
        identity = {key: data.get(key) for key in PROBE_FIELDS}
        if not data["probe_complete"]:
            # Unknown provenance must never silently merge into a compatible probe row.
            identity["unresolved_image"] = f"{job['id']}/{path.name}"
        data["probe_id"] = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()[:24]
        return data
    data.update(kind="evaluation", metadata_source="filename+config_snapshot")
    match = EVALUATION_NAME.fullmatch(path.name)
    if match is not None:
        index = int(match["index"])
        data.update(prompt_index=index, seed=int(match["seed"]))
        if match["adapter_index"] is None:
            data.update(multiplier=0.0, adapter_index=None, source_checkpoint=None)
        else:
            adapter_index = int(match["adapter_index"])
            data.update(multiplier=float(match["multiplier"]), adapter_index=adapter_index)
            if adapter_index < len(job.get("adapters", [])):
                data["source_checkpoint"] = job["adapters"][adapter_index]
    elif path.name.lower() in {"base.png", "lora.png"}:
        # Earlier paired CLI output used a promptNNN subdirectory for list prompts.
        parent_match = re.fullmatch(r"prompt([0-9]+)", path.parent.name)
        if parent_match is not None or len(prompts) == 1:
            data["prompt_index"] = int(parent_match[1]) if parent_match else 0
        if path.name.lower() == "base.png":
            data.update(multiplier=0.0, adapter_index=None, source_checkpoint=None)
        elif job.get("adapters"):
            data.update(multiplier=1.0, adapter_index=0, source_checkpoint=job["adapters"][0])
    index = data.get("prompt_index")
    if type(index) is int and 0 <= index < len(prompts) and isinstance(prompts[index], str):
        data["prompt"] = prompts[index]
    data.update(recorded)
    return data


def image_paths(job: dict) -> tuple[list[Path], bool]:
    root = safe_path(job["output_dir"])
    if job["stage"] == "cache" or not root.is_dir():
        return [], False
    candidates = []
    scanned = 0
    for directory, directories, filenames in os.walk(root, followlinks=False):
        allowed = []
        for name in directories:
            scanned += 1
            if scanned > MAX_SCANNED_FILES:
                return [path for _, path in sorted(candidates, reverse=True)[:MAX_IMAGES]], True
            try:
                safe_path(Path(directory) / name, root=root)
                allowed.append(name)
            except InputError:
                continue
        directories[:] = allowed
        for name in filenames:
            scanned += 1
            if scanned > MAX_SCANNED_FILES:
                return [path for _, path in sorted(candidates, reverse=True)[:MAX_IMAGES]], True
            accepted = PREVIEW_NAME.fullmatch(name) if job["stage"] == "train" else EVALUATION_NAME.fullmatch(name) or name.lower() in {"base.png", "lora.png"}
            if not accepted:
                continue
            try:
                path = safe_path(Path(directory) / name, root=root)
                if path.is_file():
                    candidates.append((path.stat().st_mtime_ns, path))
            except (OSError, InputError):
                continue
    return [path for _, path in sorted(candidates, reverse=True)[:MAX_IMAGES]], len(candidates) > MAX_IMAGES


def image_id(job: dict, path: Path) -> str:
    return hashlib.sha256(path.relative_to(Path(job["output_dir"])).as_posix().encode("utf-8")).hexdigest()[:32]


def gallery(job: dict) -> dict:
    job = public_job(job)
    if job["stage"] == "cache":
        return {"images": [], "truncated": False}
    metadata = metadata_index(job)
    sample = sampling_snapshot(job)
    paths, truncated = image_paths(job)
    result = []
    for path in paths:
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            width, height = struct.unpack(">II", header[16:24])
        except OSError:
            continue
        identifier = image_id(job, path)
        enriched = enrich_image(job, path, sample, metadata.get(str(path), {}))
        result.append({"id": identifier, "name": path.name, "url": f"/api/jobs/{job['id']}/images/{identifier}", "width": width, "height": height, "metadata": enriched})
    return {"images": result, "truncated": truncated}


def resolve_image(job: dict, identifier: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", identifier):
        raise KeyError("图片不存在。")
    paths, _ = image_paths(job)
    for path in paths:
        if image_id(job, path) == identifier:
            with path.open("rb") as handle:
                if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                    break
            return safe_path(path, root=Path(job["output_dir"]))
    raise KeyError("图片不存在或不属于本任务输出。")
