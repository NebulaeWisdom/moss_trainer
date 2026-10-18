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

# Types and insertion defaults follow TUTORIAL section 5, not the type of a
# particular preset. Unlisted extension keys are deliberately not constrained.
FIELD_RULES = {
    "model_dir": {"types": ["string"], "default": ""},
    "dataset.image_dir": {"types": ["string"], "default": ""},
    "dataset.caption": {"types": ["string", "null"], "default": "deepseek-tan"},
    "dataset.caption_dir": {"types": ["string", "null"], "default": None},
    "dataset.caption_format": {"types": ["string"], "default": "plain"},
    "dataset.caption_extension": {"types": ["string"], "default": ".txt"},
    "dataset.caption_weights": {"types": ["object"], "default": {"long": 1, "medium": 1, "short": 1}},
    "dataset.caption_weights.long": {"types": ["number"], "default": 1},
    "dataset.caption_weights.medium": {"types": ["number"], "default": 1},
    "dataset.caption_weights.short": {"types": ["number"], "default": 1},
    "dataset.resolutions": {"types": ["array"], "default": [512, 768], "minItems": 1, "items": {"types": ["number"], "integer": True, "default": 512, "min": 1}},
    "dataset.bucket_step": {"types": ["number"], "integer": True, "default": 64},
    "dataset.max_aspect_ratio": {"types": ["number"], "default": 2.0},
    "dataset.bucket_mode": {"types": ["string"], "default": "resize"},
    "dataset.random_crop": {"types": ["boolean"], "default": True},
    "dataset.repeats": {"types": ["number"], "integer": True, "default": 1},
    "dataset.caption_dropout": {"types": ["number"], "default": 0.0},
    "dataset.seed": {"types": ["number"], "integer": True, "default": 42},
    "text.max_sequence_length": {"types": ["number"], "integer": True, "default": 4096},
    "text.gpu_memory_gib": {"types": ["number"], "default": 10},
    "text.cpu_memory_gib": {"types": ["number"], "default": 48},
    "adapter.algo": {"types": ["string"], "default": "lora"},
    "adapter.rank": {"types": ["number"], "integer": True, "default": 8},
    "adapter.alpha": {"types": ["number"], "default": 8},
    "adapter.target_modules": {"types": ["array"], "default": ["_Block"], "minItems": 1, "items": {"types": ["string"], "default": "_Block"}},
    "adapter.dropout": {"types": ["number"], "default": 0.0},
    "adapter.kwargs": {"types": ["object"], "default": {}},
    "adapter.kwargs.use_scalar": {"types": ["boolean"], "default": False},
    "adapter.kwargs.factor": {"types": ["number"], "integer": True, "default": -1},
    "adapter.kwargs.decompose_both": {"types": ["boolean"], "default": False},
    "adapter.kwargs.unbalanced_factorization": {"types": ["boolean"], "default": False},
    "training.max_steps": {"types": ["number", "null"], "integer": True, "default": 600},
    "training.epochs": {"types": ["number", "null"], "integer": True, "default": None},
    "training.batch_size": {"types": ["number"], "integer": True, "default": 1},
    "training.gradient_accumulation": {"types": ["number"], "integer": True, "default": 1},
    "training.seed": {"types": ["number"], "integer": True, "default": 42},
    "training.gradient_checkpointing": {"types": ["boolean"], "default": True},
    "training.timestep_sampling": {"types": ["string"], "default": "uniform"},
    "training.logit_mean": {"types": ["number"], "default": 0.0},
    "training.logit_std": {"types": ["number"], "default": 1.0},
    "training.max_grad_norm": {"types": ["number"], "default": 1.0},
    "training.attention_backend": {"types": ["string"], "default": "cudnn"},
    "training.compile_pointwise": {"types": ["boolean"], "default": False},
    "training.compile_blocks": {"types": ["boolean"], "default": False},
    "training.compile_blocks_mode": {"types": ["string"], "default": "default"},
    "training.checkpoint_group_size": {"types": ["number"], "integer": True, "default": 1, "min": 1},
    "optimizer.type": {"types": ["string"], "default": "adamw"},
    "optimizer.learning_rate": {"types": ["number"], "default": 0.0001},
    "optimizer.aux_learning_rate": {"types": ["number"], "default": 0.0001, "defaultFrom": "learning_rate"},
    "optimizer.weight_decay": {"types": ["number"], "default": 0.01},
    "optimizer.kwargs": {"types": ["object"], "default": {}},
    "optimizer.kwargs.betas": {"types": ["array"], "default": [0.9, 0.999], "minItems": 2, "maxItems": 2, "items": {"types": ["number"], "default": 0.999}},
    "optimizer.kwargs.eps": {"types": ["number"], "default": 1e-8},
    "optimizer.kwargs.momentum": {"types": ["number"], "default": 0.95},
    "optimizer.kwargs.nesterov": {"types": ["boolean"], "default": True},
    "optimizer.kwargs.ns_steps": {"types": ["number"], "integer": True, "default": 5},
    "optimizer.kwargs.adjust_lr_fn": {"types": ["string", "null"], "default": None, "options": ["original", "match_rms_adamw", "spectral_unclamped"]},
    "scheduler.type": {"types": ["string"], "default": "constant"},
    "scheduler.warmup_steps": {"types": ["number"], "integer": True, "default": 0},
    "scheduler.num_training_steps": {"types": ["number"], "integer": True, "default": 1},
    "loader.num_workers": {"types": ["number"], "integer": True, "default": 2},
    "loader.pin_memory": {"types": ["boolean"], "default": True},
    "loader.prefetch_factor": {"types": ["number"], "integer": True, "default": 2},
    "loader.persistent_workers": {"types": ["boolean"], "default": True},
    "save.every_steps": {"types": ["number"], "integer": True, "default": 200},
    "save.every_epochs": {"types": ["number"], "integer": True, "default": 0},
    "save.save_state": {"types": ["boolean"], "default": True},
    "sample.preview": {"types": ["object"], "default": {"enabled": False, "every_steps": 0, "every_epochs": 0}},
    "sample.preview.enabled": {"types": ["boolean"], "default": False},
    "sample.preview.every_steps": {"types": ["number"], "integer": True, "default": 0},
    "sample.preview.every_epochs": {"types": ["number"], "integer": True, "default": 0},
    "sample.prompt": {"types": ["string", "array"], "default": "deepseek-tan", "minItems": 1, "items": {"types": ["string"], "default": ""}},
    "sample.negative_prompt": {"types": ["string"], "default": ""},
    "sample.width": {"types": ["number"], "integer": True, "default": 768},
    "sample.height": {"types": ["number"], "integer": True, "default": 512},
    "sample.num_inference_steps": {"types": ["number"], "integer": True, "default": 20, "min": 1},
    "sample.guidance_scale": {"types": ["number"], "default": 3.0},
    "sample.seed": {"types": ["number"], "integer": True, "default": 42},
}


def schema() -> dict:
    fields = {key: {"types": ["object"], "default": {}, "label": label, "description": description} for key, label, description in SECTIONS if key != "model_dir"}
    for key, rule in FIELD_RULES.items():
        label, description = FIELD_HINTS.get(key, (key.rsplit(".", 1)[-1], ""))
        fields[key] = {**rule, "label": label, "description": description}
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


def _validate_field(value: Any, path: str, rule: dict) -> None:
    kind = {str: "string", int: "number", float: "number", bool: "boolean", list: "array", dict: "object", type(None): "null"}.get(type(value))
    labels = {"string": "字符串", "number": "整数" if rule.get("integer") else "数字", "boolean": "布尔值", "array": "数组", "object": "对象", "null": "null"}
    if kind not in rule["types"] or kind == "number" and rule.get("integer") and type(value) is not int:
        raise InputError(f"{path} 必须是{'或'.join(labels[kind] for kind in rule['types'])}。")
    if kind == "number" and "min" in rule and value < rule["min"]:
        raise InputError(f"{path} 必须大于或等于 {rule['min']}。")
    if kind == "array":
        if len(value) < rule.get("minItems", 0):
            raise InputError(f"{path} 至少需要 {rule['minItems']} 项。")
        if "maxItems" in rule and len(value) > rule["maxItems"]:
            raise InputError(f"{path} 最多允许 {rule['maxItems']} 项。")
        for index, item in enumerate(value):
            _validate_field(item, f"{path}[{index}]", rule["items"])
    if kind != "null" and "options" in rule and value not in rule["options"]:
        raise InputError(f"{path} 应为: {', '.join(rule['options'])}")


def validate_config(config: Any, stage: str | None = None) -> dict:
    if not isinstance(config, dict):
        raise InputError("config 必须是 JSON 对象。")
    try:
        serialized = json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError(f"配置必须是有限、有效的 JSON: {exc}") from exc
    if len(serialized.encode("utf-8")) > MAX_JSON_BYTES:
        raise InputError("配置超过 2 MiB 上限。")
    for dotted, rule in schema()["fields"].items():
        value = config
        for key in dotted.split("."):
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            _validate_field(value, dotted, rule)
    sample = config.get("sample", {})
    if stage is None:
        return config
    required = {
        "cache": {"dataset": ["image_dir", "resolutions", "caption_dropout"], "text": ["max_sequence_length", "gpu_memory_gib", "cpu_memory_gib"]},
        "evaluation": {"training": [], "text": ["max_sequence_length", "gpu_memory_gib", "cpu_memory_gib"], "sample": ["prompt", "negative_prompt", "width", "height", "num_inference_steps", "guidance_scale", "seed"]},
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
    positive = {"cache": ["dataset.resolutions", "text.max_sequence_length"], "train": ["training.batch_size", "training.gradient_accumulation"], "evaluation": ["sample.width", "sample.height", "sample.num_inference_steps", "text.max_sequence_length"]}[stage]
    for key in positive:
        section, field = key.split(".")
        value = config[section][field]
        values = value if isinstance(value, list) else [value]
        if not values or any(type(item) is not int or item <= 0 for item in values):
            raise InputError(f"{key} 必须是正整数{'数组' if field == 'resolutions' else ''}。")
    if stage == "train" and all(config["training"][key] is None for key in ("max_steps", "epochs")):
        raise InputError("training.max_steps 与 training.epochs 至少设置一个。")
    if stage == "evaluation":
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
            validate_config(config, "evaluation")
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
