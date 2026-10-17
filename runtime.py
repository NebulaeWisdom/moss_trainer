"""Frozen model loading, DiT adapters, and MOSS-Image text conditioning."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import inspect
import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .data import cache_dataset
from .vendor.pipeline_moss_image import MOSSImageTransformer

logger = logging.getLogger(__name__)

ATTENTION_BACKENDS = ("cudnn", "auto")


def resolve_attention_backend(config: dict, *, resumed: bool) -> str:
    """Resolve ``training.attention_backend`` in place and return the setting.

    A new run defaults to cuDNN priority; a resumed run whose stored config
    predates the setting keeps the runtime default so old checkpoints replay the
    kernel choice they were trained with.
    """
    training = config["training"]
    if "attention_backend" not in training:
        training["attention_backend"] = "auto" if resumed else "cudnn"
        logger.info(
            "training.attention_backend not configured; resolved to %r for a %s run",
            training["attention_backend"],
            "resumed" if resumed else "new",
        )
    backend = training["attention_backend"]
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"training.attention_backend must be one of {ATTENTION_BACKENDS}, got {backend!r}"
        )
    return backend


def resolve_compile_pointwise(config: dict) -> bool:
    """Persist the opt-in compiler setting for both new and resumed runs."""
    training = config["training"]
    enabled = training.setdefault("compile_pointwise", False)
    if type(enabled) is not bool:
        raise ValueError(f"training.compile_pointwise must be a boolean, got {enabled!r}")
    if enabled:
        from .kernels import validate_pointwise_support

        validate_pointwise_support()
    logger.info("resolved training.compile_pointwise=%s", enabled)
    return enabled


def resolve_checkpoint_group_size(config: dict) -> int:
    """Persist group one for new runs and stored configs predating grouping."""
    training = config["training"]
    group_size = training.setdefault("checkpoint_group_size", 1)
    if type(group_size) is not int or group_size < 1:
        raise ValueError(
            f"training.checkpoint_group_size must be a positive integer, got {group_size!r}"
        )
    return group_size


@contextlib.contextmanager
def training_kernels(config: dict) -> Iterator[None]:
    """Scope attention priority and optional pointwise compilation to training.

    The caller includes backward/recomputation and DiT previews, but excludes
    text-encoder loading/encoding. All patched functions and backend priorities
    are restored on exit, including when training raises.
    """
    backend = config["training"]["attention_backend"]
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"training.attention_backend must be one of {ATTENTION_BACKENDS}, got {backend!r}"
        )
    compile_pointwise = config["training"].get("compile_pointwise", False)
    if type(compile_pointwise) is not bool:
        raise ValueError(
            f"training.compile_pointwise must be a boolean, got {compile_pointwise!r}"
        )
    with contextlib.ExitStack() as stack:
        if backend == "cudnn":
            # Refuse rather than guess a fallback on older torch releases.
            from torch.nn.attention import SDPBackend, sdpa_kernel

            if "set_priority" not in inspect.signature(sdpa_kernel).parameters:
                raise RuntimeError(
                    "training.attention_backend='cudnn' requires torch.nn.attention.sdpa_kernel(set_priority=True)"
                )
            backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
            stack.enter_context(sdpa_kernel(backends, set_priority=True))
        if compile_pointwise:
            from .kernels import pointwise_kernels

            stack.enter_context(pointwise_kernels())
        yield


def load_dit(config: dict, checkpointing: bool = False) -> MOSSImageTransformer:
    """Load the frozen BF16 DiT; the training caller explicitly selects train mode."""
    group_size = resolve_checkpoint_group_size(config)
    dit = MOSSImageTransformer.from_pretrained(
        Path(config["model_dir"]) / "transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    dit.requires_grad_(False)
    dit.eval()
    dit.to("cuda")
    dit._configure_checkpointing(checkpointing, group_size)
    return dit


def load_vae(config: dict, device: str = "cuda") -> AutoencoderKLFlux2:
    """Keep the frozen Flux2 VAE in FP32 for both encoding and decoding."""
    vae = AutoencoderKLFlux2.from_pretrained(
        Path(config["model_dir"]) / "vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.requires_grad_(False)
    vae.eval()
    return vae.to(device=device, dtype=torch.float32)


def normalize_latents(raw: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Invert the official decoder's unpatchify and batch-norm affine transform."""
    batch, channels, height, width = raw.shape
    height //= 2
    width //= 2
    z = raw.float().reshape(batch, channels, height, 2, width, 2)
    z = z.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height, width)
    mean = stats["mean"].to(device=z.device, dtype=torch.float32).view(1, -1, 1, 1)
    var = stats["var"].to(device=z.device, dtype=torch.float32).view(1, -1, 1, 1)
    eps = stats["eps"].to(device=z.device, dtype=torch.float32)
    return (z - mean) / torch.sqrt(var + eps)


def create_adapter(dit: MOSSImageTransformer, config: dict):
    """Apply the installed LyCORIS distribution to configured DiT blocks.

    ``adapter.kwargs`` is forwarded to ``create_lycoris`` verbatim (real LyCORIS
    options such as ``factor``/``decompose_both``); the preset only targets DiT
    blocks, so no text-encoder weights are ever adapted.
    """
    from lycoris.config import PRESET
    from lycoris.wrapper import create_lycoris

    adapter = config["adapter"]
    # create_lycoris always applies its preset; a prior apply_preset alone is lost.
    PRESET["moss_image_dit"] = {
        "target_module": adapter["target_modules"],
        "target_name": [],
        "enable_conv": False,
        "lora_prefix": "lora_unet",
    }
    network = create_lycoris(
        dit,
        multiplier=1.0,
        linear_dim=adapter["rank"],
        linear_alpha=adapter["alpha"],
        algo=adapter["algo"],
        dropout=adapter["dropout"],
        preset="moss_image_dit",
        bypass_mode=True,
        **adapter.get("kwargs", {}),
    )
    network.apply_to()
    network.to(device="cuda", dtype=torch.float32)
    network.requires_grad_(True)
    return network


@torch.no_grad()
def encode_text(config: dict, captions: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream unpadded CPU embeddings, releasing the frozen TE on completion."""
    text_config = config["text"]
    model_dir = Path(config["model_dir"])
    tokenizer = None
    text_encoder = None
    started = time.perf_counter()
    logger.info("encoding %d texts with frozen text encoder", len(captions))
    try:
        load_started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
        text_encoder = AutoModel.from_pretrained(
            model_dir / "text_encoder",
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={
                0: f"{text_config['gpu_memory_gib']}GiB",
                "cpu": f"{text_config['cpu_memory_gib']}GiB",
            },
        )
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        logger.info("loaded frozen text encoder: load_seconds=%.3f", time.perf_counter() - load_started)
        for caption in tqdm(
            captions, file=sys.stdout, unit="caption", desc="Encode text", dynamic_ncols=True, mininterval=0.5
        ):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": caption}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=text_config["max_sequence_length"],
            ).to("cuda")
            output = text_encoder(**inputs, use_cache=False, return_dict=True)
            embedding = output.last_hidden_state[0].detach().cpu().contiguous()
            del output, inputs
            yield caption, embedding
            del embedding
        logger.info("encoded %d texts: text_encoding_seconds=%.3f", len(captions), time.perf_counter() - started)
    finally:
        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def _cache_text(config: dict, captions: list[str], cache_dir: Path) -> None:
    text_dir = cache_dir / "text"
    text_dir.mkdir()
    manifest = {}
    for caption, embedding in encode_text(config, captions):
        text_path = text_dir / f"{hashlib.sha256(caption.encode('utf-8')).hexdigest()}.safetensors"
        save_file({"embedding": embedding}, str(text_path))
        manifest[caption] = str(text_path.resolve())
        del embedding
    with (cache_dir / "text_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen VAE latents and text conditioning for MOSS-Image.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    cache_dir = args.cache_dir.resolve()

    # data.py owns the empty-directory guard: write latents before stats or text.
    vae = load_vae(config)
    records = cache_dataset(config["dataset"], vae, cache_dir, "cuda", torch.float32)
    save_file(
        {
            "mean": vae.bn.running_mean.detach().float().cpu().contiguous(),
            "var": vae.bn.running_var.detach().float().cpu().contiguous(),
            "eps": torch.tensor(vae.config.batch_norm_eps, dtype=torch.float32),
        },
        str(cache_dir / "latent_stats.safetensors"),
    )
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    if config["dataset"].get("caption_format", "plain") == "moss_official":
        captions = [record["caption"][level] for record in records for level in ("long", "medium", "short")]
    else:
        captions = [record["caption"] for record in records]
    if float(config["dataset"]["caption_dropout"]) > 0:
        captions.append("")
    captions = list(dict.fromkeys(captions))
    _cache_text(config, captions, cache_dir)
    logger.info("cached %d image records and %d distinct captions into %s", len(records), len(captions), cache_dir)


if __name__ == "__main__":
    main()
