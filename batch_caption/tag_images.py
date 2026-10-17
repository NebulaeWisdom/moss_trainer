#!/usr/bin/env python3
"""Caption images recursively with an OpenAI-compatible vision endpoint."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from openai import OpenAI, OpenAIError
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
DEFAULTS = {
    "reasoning_effort": "high",
    "max_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
    "concurrency": 3,
    "retries": 2,
    "timeout": 300,
    "input_dir": "../test_image",
    "output_dir": "output",
    "max_image_size": 2048,
    "prompt_file": "prompt.txt",
    "reference": "",
}


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().with_name("config.yaml"),
        help="YAML configuration (default: beside this script)",
    )
    parser.add_argument("--input-dir", type=Path, help="input override, relative to the current directory")
    parser.add_argument("--output-dir", type=Path, help="output override, relative to the current directory")
    parser.add_argument("--limit", type=positive_int, help="take the first N sorted images before resume checks")
    parser.add_argument("--force", action="store_true", help="regenerate even when output JSON already exists")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> tuple[dict, Path, Path, str]:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        supplied = yaml.safe_load(handle)
    if not isinstance(supplied, dict):
        raise ValueError("configuration must be a YAML mapping")
    config = {**DEFAULTS, **supplied}
    for key in ("base_url", "api_key", "model", "reasoning_effort", "input_dir", "output_dir", "prompt_file"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(config["reference"], str):
        raise ValueError("reference must be a string")
    for key, minimum in (("max_tokens", 1), ("concurrency", 1), ("retries", 0), ("max_image_size", 0)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("temperature", "top_p", "timeout"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not 0 <= config["temperature"] <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if not 0 <= config["top_p"] <= 1:
        raise ValueError("top_p must be between 0 and 1")
    if config["timeout"] <= 0:
        raise ValueError("timeout must be greater than zero")

    base = config_path.parent
    input_dir = (args.input_dir.expanduser().resolve() if args.input_dir is not None
                 else (base / Path(config["input_dir"]).expanduser()).resolve())
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None
                  else (base / Path(config["output_dir"]).expanduser()).resolve())
    prompt_path = (base / Path(config["prompt_file"]).expanduser()).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist or is not a directory: {input_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output directory is not a directory: {output_dir}")
    prompt = prompt_path.read_text(encoding="utf-8-sig").replace("{{reference}}", config["reference"])
    if not prompt.strip():
        raise ValueError(f"prompt is empty: {prompt_path}")
    return config, input_dir, output_dir, prompt


def image_data_url(path: Path, max_image_size: int) -> str:
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source)
        try:
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            converted = image.convert("RGBA" if has_alpha else "RGB")
        finally:
            image.close()
    with converted:
        if max_image_size:
            converted.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
        with io.BytesIO() as buffer:
            if has_alpha:
                converted.save(buffer, format="PNG")
                mime = "image/png"
            else:
                converted.save(buffer, format="JPEG", quality=90)
                mime = "image/jpeg"
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def validate_caption(result: object) -> dict:
    if not isinstance(result, dict) or set(result) != {"caption", "watermark", "borders"}:
        raise ValueError("result must contain exactly caption, watermark, and borders")
    caption = result["caption"]
    if not isinstance(caption, dict) or set(caption) != {"long", "medium", "short"}:
        raise ValueError("caption must contain exactly long, medium, and short")
    for name, value in caption.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"caption.{name} must be a non-empty string")
    if not isinstance(result["watermark"], list):
        raise ValueError("watermark must be a list")
    for index, mark in enumerate(result["watermark"]):
        if not isinstance(mark, dict) or set(mark) != {"box_2d", "label"}:
            raise ValueError(f"watermark[{index}] must contain exactly box_2d and label")
        box = mark["box_2d"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"watermark[{index}].box_2d must contain four coordinates")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not 0 <= value <= 1000 for value in box):
            raise ValueError(f"watermark[{index}].box_2d coordinates must be finite numbers in 0..1000")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"watermark[{index}].box_2d must have y_min < y_max and x_min < x_max")
        if not isinstance(mark["label"], str) or not mark["label"].strip():
            raise ValueError(f"watermark[{index}].label must be a non-empty string")
    borders = result["borders"]
    if not isinstance(borders, list) or not borders:
        raise ValueError("borders must be a non-empty list")
    if any(not isinstance(value, str) or value not in {"none", "left", "right", "up", "down"}
           for value in borders):
        raise ValueError("borders may contain only none, left, right, up, or down")
    if len(set(borders)) != len(borders) or ("none" in borders and borders != ["none"]):
        raise ValueError("borders must not repeat directions or combine none with directions")
    return result


def request_caption(client: OpenAI, config: dict, prompt: str, data_url: str) -> dict:
    response = client.chat.completions.with_raw_response.create(
        model=config["model"],
        reasoning_effort=config["reasoning_effort"],
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    ).http_response.json()
    if isinstance(response, dict) and "choices" not in response:
        response = response.get("data")
    if not isinstance(response, dict):
        raise ValueError("API response must contain a chat completion object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("API response has no usable choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("API response message.content is empty or not text")
    return validate_caption(json.loads(content))


def has_parseable_json(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            json.load(handle)
        return True
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return False


def save_json_atomic(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix="caption-", suffix=".json.tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                print(f"WARNING: cannot remove temporary file {temporary}: {exc}", file=sys.stderr)


def process_image(
    path: Path, output_path: Path, client: OpenAI, config: dict, prompt: str,
) -> tuple[str, str]:
    try:
        data_url = image_data_url(path, config["max_image_size"])
        for attempt in range(config["retries"] + 1):
            try:
                result = request_caption(client, config, prompt, data_url)
                break
            except (OpenAIError, ValueError) as exc:
                if attempt == config["retries"]:
                    return "failed", f"after {attempt + 1} attempt(s): {exc}"
                time.sleep(min(2 ** attempt, 8))
        save_json_atomic(output_path, result)
        return "success", ""
    except Exception as exc:
        return "failed", str(exc)


def print_counts(counts: dict[str, int]) -> None:
    print(f"Success: {counts['success']} | Skipped: {counts['skipped']} | Failed: {counts['failed']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config, input_dir, output_dir, prompt = load_config(args)
        images = sorted(
            (path for path in input_dir.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(input_dir).as_posix(),
        )
    except (OSError, ValueError, UnicodeError, yaml.YAMLError) as exc:
        print(f"Configuration/input error: {exc}", file=sys.stderr)
        return 2

    counts = {"success": 0, "skipped": 0, "failed": 0}
    if not images:
        print(f"No supported images found in {input_dir}")
        print_counts(counts)
        return 0

    # Check every source before limiting: even a selected/unselected pair can collide.
    destinations: dict[str, list[Path]] = {}
    for path in images:
        relative_output = path.relative_to(input_dir).with_suffix(".json")
        key = os.path.normcase(str(output_dir / relative_output))
        destinations.setdefault(key, []).append(path)
    selected = images[:args.limit] if args.limit is not None else images
    pending = []
    for path in selected:
        output_path = output_dir / path.relative_to(input_dir).with_suffix(".json")
        colliding = destinations[os.path.normcase(str(output_path))]
        if len(colliding) > 1:
            names = ", ".join(str(other.relative_to(input_dir)) for other in colliding)
            print(f"FAILED {path}: output collision at {output_path} ({names})", file=sys.stderr)
            counts["failed"] += 1
            continue
        try:
            if not args.force and has_parseable_json(output_path):
                counts["skipped"] += 1
                continue
        except OSError as exc:
            print(f"FAILED {path}: cannot read existing output: {exc}", file=sys.stderr)
            counts["failed"] += 1
            continue
        pending.append((path, output_path))

    if not pending:
        print_counts(counts)
        return 1 if counts["failed"] else 0

    try:
        with OpenAI(
            base_url=config["base_url"], api_key=config["api_key"],
            timeout=config["timeout"], max_retries=0,
        ) as client:
            executor = ThreadPoolExecutor(max_workers=config["concurrency"])
            futures = {
                executor.submit(process_image, path, output_path, client, config, prompt): path
                for path, output_path in pending
            }
            for future in as_completed(futures):
                status, reason = future.result()
                counts[status] += 1
                if status == "failed":
                    print(f"FAILED {futures[future]}: {reason}", file=sys.stderr)
                else:
                    print(f"OK {futures[future].relative_to(input_dir)}")
    except (OpenAIError, OSError, ValueError) as exc:
        print(f"Client error: {exc}", file=sys.stderr)
        return 2

    print_counts(counts)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Plain exit would join the pool's non-daemon workers and hang on in-flight
        # requests; leave the process now.
        os._exit(130)
