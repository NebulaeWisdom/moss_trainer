"""JSON configuration revisions and workspace-confined output paths."""
from __future__ import annotations

import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
STATE_DIR = PROJECT_DIR / ".webui"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "runs" / "webui"
MAX_JSON_BYTES = 2 * 1024 * 1024


class InputError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex


def safe_path(value: str | Path, root: Path = WORKSPACE_DIR) -> Path:
    """Reject traversal, symlinks and Windows junctions, including existing ancestors."""
    raw = Path(value)
    if ".." in raw.parts:
        raise InputError("路径不能包含 .. / Paths may not contain '..'.")
    if os.name == "nt" and (raw.drive and not raw.is_absolute() or any(":" in part for part in raw.parts if part != raw.anchor)):
        raise InputError("不允许相对盘符路径或 NTFS 备用数据流。")
    path = raw if raw.is_absolute() else WORKSPACE_DIR / raw
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise InputError(f"输出和状态路径必须位于工作区内: {root}") from exc
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise InputError(f"不允许符号链接或目录联接: {current}")
    if path.resolve().is_relative_to(root.resolve()) is False:
        raise InputError("路径离开了工作区。")
    return path


def read_json(path: Path, limit: int = MAX_JSON_BYTES) -> Any:
    with path.open("rb") as handle:
        content = handle.read(limit + 1)
    if len(content) > limit:
        raise InputError(f"JSON 文件过大 (上限 {limit} bytes): {path.name}")
    try:
        return json.loads(content.decode("utf-8-sig"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, ValueError) as exc:
        raise InputError(f"无效 JSON 文件 {path.name}: {exc}") from exc


def write_new_json(path: Path, value: Any) -> None:
    path = safe_path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def input_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{label}: 请填写本地路径。")
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE_DIR / path
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise InputError(f"{label}: {'目录' if directory else '文件'}不存在: {path}")
    return path


SECTIONS = [
    ("model_dir", "基础模型", "离线模型目录；只读，不下载模型。"),
    ("dataset", "数据与分桶", "图像、描述与分辨率；几何或文本编码设置改变后需要新缓存。"),
    ("text", "文本编码", "冻结文本编码器的长度与显存/内存预算。"),
    ("adapter", "适配器", "LyCORIS LoRA / LoKr 参数；采样时必须匹配权重。"),
    ("training", "训练", "单 GPU 优化步、累积、内核与随机种子。"),
    ("optimizer", "优化器", "AdamW / Muon；kwargs 原样传递。"),
    ("scheduler", "学习率调度", "沿用 diffusers 调度器；总步数由训练计算。"),
    ("loader", "数据加载", "CPU worker 和预取设置。"),
    ("save", "检查点保存", "按步数或完整 epoch 保存；0 表示禁用该触发条件。"),
    ("sample", "采样与预览", "prompt 支持字符串或字符串数组；负向提示词为共享字符串。"),
]

# Hints describe existing trainer fields; unknown JSON keys remain untouched.
FIELD_HINTS = {
    "model_dir": ("模型目录", "包含 transformer、vae、text_encoder、tokenizer 的本地模型目录。"),
    "dataset.image_dir": ("图像目录", "递归读取训练图像。"),
    "dataset.caption": ("固定描述", "null 使用图像对应描述文件；字符串为固定描述。"),
    "dataset.caption_dir": ("描述目录", "null 使用图像所在目录。"),
    "dataset.caption_format": ("描述格式", "plain 或 MOSS 官方三级 JSON 描述。"),
    "dataset.caption_extension": ("描述扩展名", "如 .txt / .json。"),
    "dataset.caption_weights": ("描述抽样权重", "long / medium / short 的相对权重。"),
    "dataset.resolutions": ("分辨率列表", "缓存分桶的目标分辨率。"),
    "dataset.bucket_step": ("分桶步长", "分桶尺寸的网格步长。"),
    "dataset.max_aspect_ratio": ("最大宽高比", "限制极端宽高比。"),
    "dataset.bucket_mode": ("分桶模式", "resize 或 no_upscale。"),
    "dataset.random_crop": ("随机裁剪", "缓存时按随机种子裁剪。"),
    "dataset.repeats": ("数据重复次数", "虚拟数据重复；不需重新缓存。"),
    "dataset.caption_dropout": ("描述丢弃概率", "训练时使用空描述的概率；缓存须包含空文本编码。"),
    "dataset.seed": ("缓存种子", "控制缓存几何处理的随机性。"),
    "text.max_sequence_length": ("最大文本长度", "文本 token 上限；改变后需新缓存。"),
    "text.gpu_memory_gib": ("文本编码 GPU 预算", "单位 GiB。"),
    "text.cpu_memory_gib": ("文本编码 CPU 预算", "单位 GiB。"),
    "adapter.algo": ("适配器算法", "LyCORIS 算法名称。"),
    "adapter.rank": ("Rank", "低秩维度。"),
    "adapter.alpha": ("Alpha", "适配器缩放参数。"),
    "adapter.target_modules": ("目标模块", "沿用模型模块名，如 _Block。"),
    "adapter.dropout": ("适配器 Dropout", "传递给 LyCORIS。"),
    "adapter.kwargs": ("算法附加参数", "原样传递，如 LoKr factor/use_scalar。"),
    "training.max_steps": ("最大优化步", "与 epochs 同时设置时取先到达者；null 不限制。"),
    "training.epochs": ("Epoch 上限", "null 不限制；至少设置一个训练上限。"),
    "training.batch_size": ("批大小", "同桶 batch 的图像数。"),
    "training.gradient_accumulation": ("梯度累积", "一次优化更新包含的 microbatch 数。"),
    "training.seed": ("训练种子", "控制随机数与噪声生成。"),
    "training.gradient_checkpointing": ("梯度检查点", "用重计算降低激活显存。"),
    "training.timestep_sampling": ("时间步采样", "uniform 或 logit_normal。"),
    "training.logit_mean": ("Logit 均值", "logit_normal 的均值。"),
    "training.logit_std": ("Logit 标准差", "logit_normal 的标准差。"),
    "training.max_grad_norm": ("梯度裁剪", "优化前裁剪梯度范数。"),
    "training.attention_backend": ("注意力后端", "沿用训练器支持的注意力实现。"),
    "training.compile_pointwise": ("编译逐点算子", "使用现有训练器编译路径。"),
    "training.compile_blocks": ("编译块", "仅 LoRA，且需开启 compile_pointwise。"),
    "training.compile_blocks_mode": ("块编译模式", "沿用现有训练器模式。"),
    "training.checkpoint_group_size": ("检查点分组", "每组连续 block 数；至少 1。"),
    "optimizer.type": ("优化器", "adamw 或 muon。"),
    "optimizer.learning_rate": ("学习率", "主优化器初始学习率。"),
    "optimizer.aux_learning_rate": ("辅助学习率", "Muon 非矩阵参数 AdamW 学习率。"),
    "optimizer.weight_decay": ("权重衰减", "传递给优化器。"),
    "optimizer.kwargs": ("优化器附加参数", "原样传递给 PyTorch 优化器。"),
    "scheduler.type": ("调度器", "diffusers scheduler 名称。"),
    "scheduler.warmup_steps": ("预热步数", "constant 配合非零预热会使用 constant_with_warmup。"),
    "scheduler.num_training_steps": ("已解析总步数", "新训练自动计算；恢复训练从检查点还原。"),
    "loader.num_workers": ("加载进程数", "0 在训练进程中加载。"),
    "loader.pin_memory": ("锁页内存", "用于 CPU 到 GPU 传输。"),
    "loader.prefetch_factor": ("预取批数", "每个 worker 的预取批数。"),
    "loader.persistent_workers": ("持久 worker", "跨 epoch 保留加载进程。"),
    "save.every_steps": ("每 N 步保存", "0 关闭按步保存。"),
    "save.every_epochs": ("每 N 个 Epoch 保存", "0 关闭按 epoch 保存。"),
    "save.save_state": ("保存完整状态", "含优化器与随机状态；恢复仅信任本机生成的检查点。"),
    "sample.preview.enabled": ("训练预览", "在训练进程中生成预览，不新开模型进程。"),
    "sample.preview.every_steps": ("每 N 步预览", "0 关闭按步预览。"),
    "sample.preview.every_epochs": ("每 N 个 Epoch 预览", "0 关闭按 epoch 预览。"),
    "sample.prompt": ("正向提示词", "字符串或字符串数组；多个提示词分别生成图片。"),
    "sample.negative_prompt": ("负向提示词", "全部提示词共享一个字符串。"),
    "sample.width": ("宽度", "像素，沿用模型要求的 16 倍数。"),
    "sample.height": ("高度", "像素，沿用模型要求的 16 倍数。"),
    "sample.num_inference_steps": ("采样步数", "扩散采样迭代次数。"),
    "sample.guidance_scale": ("CFG", "提示词引导强度。"),
    "sample.seed": ("采样种子", "基模与适配器比较使用相同种子。"),
}


def schema() -> dict:
    fields = {key: {"label": label, "description": description} for key, (label, description) in FIELD_HINTS.items()}
    for key, options in {
        "dataset.caption_format": ["plain", "moss_official"],
        "dataset.bucket_mode": ["resize", "no_upscale"],
        "optimizer.type": ["adamw", "muon"],
        "training.timestep_sampling": ["uniform", "logit_normal"],
        "training.attention_backend": ["cudnn", "auto"],
        "training.compile_blocks_mode": ["default", "max-autotune-no-cudagraphs"],
    }.items():
        fields[key]["options"] = options
    return {"sections": [{"key": key, "label": label, "description": description} for key, label, description in SECTIONS], "fields": fields}


def validate_config(config: Any, stage: str | None = None) -> dict:
    if not isinstance(config, dict):
        raise InputError("config 必须是 JSON 对象。")
    try:
        serialized = json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError(f"配置必须是有限、有效的 JSON: {exc}") from exc
    if len(serialized.encode("utf-8")) > MAX_JSON_BYTES:
        raise InputError("配置超过 2 MiB 上限。")
    for key, _, _ in SECTIONS:
        if key in config and key != "model_dir" and not isinstance(config[key], dict):
            raise InputError(f"{key} 必须是对象。")
    if "model_dir" in config and not isinstance(config["model_dir"], str):
        raise InputError("model_dir 必须是字符串路径。")
    sample = config.get("sample", {})
    if "prompt" in sample and not (isinstance(sample["prompt"], str) or isinstance(sample["prompt"], list) and sample["prompt"] and all(isinstance(item, str) for item in sample["prompt"])):
        raise InputError("sample.prompt 必须是字符串或非空字符串数组。")
    if "negative_prompt" in sample and not isinstance(sample["negative_prompt"], str):
        raise InputError("sample.negative_prompt 必须是字符串。")
    for section, key in (("adapter", "kwargs"), ("optimizer", "kwargs"), ("sample", "preview"), ("dataset", "caption_weights")):
        if key in config.get(section, {}) and not isinstance(config[section][key], dict):
            raise InputError(f"{section}.{key} 必须是对象。")
    integer_fields = {
        "dataset": ("bucket_step", "repeats", "seed"),
        "text": ("max_sequence_length",),
        "adapter": ("rank",),
        "training": ("max_steps", "epochs", "batch_size", "gradient_accumulation", "seed", "checkpoint_group_size"),
        "scheduler": ("warmup_steps", "num_training_steps"),
        "loader": ("num_workers", "prefetch_factor"),
        "save": ("every_steps", "every_epochs"),
        "sample": ("width", "height", "num_inference_steps", "seed"),
    }
    number_fields = {
        "dataset": ("max_aspect_ratio", "caption_dropout"),
        "text": ("gpu_memory_gib", "cpu_memory_gib"),
        "adapter": ("alpha", "dropout"),
        "training": ("logit_mean", "logit_std", "max_grad_norm"),
        "optimizer": ("learning_rate", "aux_learning_rate", "weight_decay"),
        "sample": ("guidance_scale",),
    }
    boolean_fields = {
        "dataset": ("random_crop",),
        "training": ("gradient_checkpointing", "compile_pointwise", "compile_blocks"),
        "loader": ("pin_memory", "persistent_workers"),
        "save": ("save_state",),
    }
    for groups, types, label in ((integer_fields, (int,), "整数"), (number_fields, (int, float), "数字"), (boolean_fields, (bool,), "布尔值")):
        for section, keys in groups.items():
            for key in keys:
                if key not in config.get(section, {}):
                    continue
                value = config[section][key]
                if value is None and section == "training" and key in ("max_steps", "epochs"):
                    continue
                if type(value) not in types:
                    raise InputError(f"{section}.{key} 必须是{label}。")
                if section == "training" and key == "checkpoint_group_size" and value < 1:
                    raise InputError("training.checkpoint_group_size 必须是正整数。")
    for key in ("caption", "caption_dir"):
        value = config.get("dataset", {}).get(key)
        if value is not None and not isinstance(value, str):
            raise InputError(f"dataset.{key} 必须是字符串或 null。")
    for section, field, item_type in (("dataset", "resolutions", int), ("adapter", "target_modules", str)):
        if field in config.get(section, {}):
            value = config[section][field]
            if not isinstance(value, list) or not value or any(type(item) is not item_type for item in value):
                raise InputError(f"{section}.{field} 必须是非空{'整数' if item_type is int else '字符串'}数组。")
    preview = sample.get("preview", {})
    for key, expected in (("enabled", bool), ("every_steps", int), ("every_epochs", int)):
        if key in preview and type(preview[key]) is not expected:
            raise InputError(f"sample.preview.{key} 类型无效。")
    for dotted, hint in schema()["fields"].items():
        if "options" not in hint:
            continue
        section, key = dotted.split(".", 1)
        if key in config.get(section, {}) and config[section][key] not in hint["options"]:
            raise InputError(f"{dotted} 应为: {', '.join(hint['options'])}")
    if stage is None:
        return config
    required = {
        "cache": {"dataset": ["image_dir", "resolutions", "caption_dropout"], "text": ["max_sequence_length", "gpu_memory_gib", "cpu_memory_gib"]},
        "sample": {"training": [], "text": ["max_sequence_length", "gpu_memory_gib", "cpu_memory_gib"], "sample": ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale", "seed"]},
        "train": {"dataset": [], "adapter": ["algo", "rank", "alpha", "target_modules", "dropout"], "training": ["max_steps", "epochs", "batch_size", "gradient_accumulation", "seed", "gradient_checkpointing", "timestep_sampling", "logit_mean", "logit_std", "max_grad_norm"], "optimizer": ["type", "learning_rate", "weight_decay", "kwargs"], "scheduler": ["type", "warmup_steps"], "loader": [], "save": ["every_steps", "every_epochs", "save_state"]},
    }[stage]
    missing_sections = [section for section in required if section not in config]
    if missing_sections:
        raise InputError("配置缺少必需对象: " + ", ".join(missing_sections))
    missing = [f"{section}.{key}" for section, keys in required.items() for key in keys if key not in config.get(section, {})]
    if "model_dir" not in config:
        missing.insert(0, "model_dir")
    if missing:
        raise InputError("配置缺少必需字段: " + ", ".join(missing))
    input_path(config["model_dir"], "model_dir", directory=True)
    if stage == "cache":
        input_path(config["dataset"]["image_dir"], "dataset.image_dir", directory=True)
        if config["dataset"].get("caption_dir"):
            input_path(config["dataset"]["caption_dir"], "dataset.caption_dir", directory=True)
    positive = {"cache": ["dataset.resolutions", "text.max_sequence_length"], "train": ["training.batch_size", "training.gradient_accumulation"], "sample": ["sample.width", "sample.height", "sample.num_inference_steps", "text.max_sequence_length"]}[stage]
    for key in positive:
        section, field = key.split(".")
        value = config[section][field]
        values = value if isinstance(value, list) else [value]
        if not values or any(type(item) is not int or item <= 0 for item in values):
            raise InputError(f"{key} 必须是正整数{'数组' if field == 'resolutions' else ''}。")
    if stage == "train" and all(config["training"][key] is None for key in ("max_steps", "epochs")):
        raise InputError("training.max_steps 与 training.epochs 至少设置一个。")
    if stage == "sample":
        for key in ("width", "height"):
            if sample[key] % 16:
                raise InputError(f"sample.{key} 必须为 16 的正整数倍。")
        if type(sample["guidance_scale"]) not in (int, float) or sample["guidance_scale"] < 0:
            raise InputError("sample.guidance_scale 必须为非负数。")
    if stage == "train":
        training = config["training"]
        for key in ("max_steps", "epochs"):
            if training[key] is not None and (type(training[key]) is not int or training[key] <= 0):
                raise InputError(f"training.{key} 必须为正整数或 null。")
        if training.get("compile_blocks"):
            if not training.get("compile_pointwise") or config["adapter"].get("algo") != "lora":
                raise InputError("compile_blocks 需要 compile_pointwise=true 且 adapter.algo=lora。")
        preview = config.get("sample", {}).get("preview", {})
        if preview.get("enabled") and (preview.get("every_steps", 0) or preview.get("every_epochs", 0)):
            missing = [key for key in ("every_steps", "every_epochs") if key not in preview]
            if missing:
                raise InputError("sample.preview 缺少触发字段: " + ", ".join(missing))
            validate_config(config, "sample")
    return config


class ConfigStore:
    def __init__(self) -> None:
        safe_path(STATE_DIR / "configs").mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        result = []
        for directory, prefix in ((PROJECT_DIR / "configs", "builtin"), (STATE_DIR / "configs", "revision")):
            for path in sorted(safe_path(directory).glob("*.json")):
                try:
                    safe_path(path)
                except InputError:
                    continue
                result.append({"id": f"{prefix}:{path.stem}", "name": path.stem, "path": str(path)})
        return result

    def get(self, identifier: str) -> dict:
        entry = next((item for item in self.entries() if item["id"] == identifier), None)
        if entry is None:
            raise KeyError("配置不存在。")
        return {**entry, "config": validate_config(read_json(safe_path(entry["path"])))}

    def save(self, name: str, config: dict) -> dict:
        validate_config(config)
        slug = re.sub(r"[^\w-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")[:64] or "config"
        path = safe_path(STATE_DIR / "configs" / f"{slug}-{new_id()}.json")
        write_new_json(path, config)
        return {"id": f"revision:{path.stem}", "name": path.stem, "path": str(path), "config": config}
