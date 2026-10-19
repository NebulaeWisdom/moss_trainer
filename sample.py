"""Official MOSS-Image sampling: reusable sampler/decode helpers, the live training
preview, and the paired base/adapter CLI with identical noise."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .launch import resolve_model_paths
from .runtime import create_adapter, encode_text, load_dit, load_vae
from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler

logger = logging.getLogger(__name__)


def create_sample_scheduler(sample: dict) -> MOSSImageScheduler:
    """Use the official nearest-grid Euler scheduler with optional config overrides."""
    return MOSSImageScheduler(**sample.get("scheduler", {}))


def decode_latents(vae, latents: torch.Tensor, image_processor, output_type: str = "pil"):
    """Official Flux2 FP32 decode: undo the BN affine, undo the 2x2 latent patchify, decode."""
    z = latents.to(dtype=vae.dtype)
    z = z * (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    z = z + vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    batch, channels, height, width = z.shape
    z = z.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3)
    z = z.reshape(batch, channels // 4, 2 * height, 2 * width)
    decoded = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    return image_processor.postprocess(decoded, output_type=output_type)


def sample_latents(
    pipe: MOSSImagePipeline,
    sample: dict,
    positive: torch.Tensor,
    positive_mask: torch.Tensor,
    negative: torch.Tensor,
    negative_mask: torch.Tensor,
    generator: torch.Generator,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the official sampler once and return raw FP32 latents, without VAE decoding.

    ``autocast_dtype`` wraps only the denoising passes; latents stay FP32 either way.
    """
    context = nullcontext() if autocast_dtype is None else torch.autocast("cuda", dtype=autocast_dtype)
    with context:
        return pipe(
            prompt_embeds=positive,
            prompt_attention_mask=positive_mask,
            negative_prompt_embeds=negative,
            negative_prompt_attention_mask=negative_mask,
            width=sample["width"],
            height=sample["height"],
            num_inference_steps=sample["num_inference_steps"],
            guidance_scale=sample["guidance_scale"],
            generator=generator,
            output_type="latent",
            **{
                key: sample[key]
                for key in ("uncond_skip_layers", "slg_layers", "slg_scale", "slg_start", "slg_stop")
                if key in sample
            },
        ).images


@torch.no_grad()
def generate_preview(
    dit, network, config: dict, conditioning: dict, output_path: Path
) -> dict:
    """Sample adapter previews from CPU conditioning with one temporary FP32 VAE.

    String prompts retain ``output_path``; lists add a zero-based prompt suffix.
    Training state is untouched: the FP32 adapter dtype/optimizer/multiplier stay
    as-is, the previous train/eval modes are restored, and all previews run under
    one ``torch.random.fork_rng`` with Python/numpy RNG state restored. The DiT
    denoises under BF16 autocast; the VAE decode runs FP32 outside autocast.
    """
    sample = config["sample"]
    single_prompt = isinstance(sample["prompt"], str)
    prompts = [sample["prompt"]] if single_prompt else sample["prompt"]
    output_path = Path(output_path)
    started = time.perf_counter()
    outputs = []

    vae = None
    pipe = None
    dit_was_training = dit.training
    network_was_training = network.training
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    dit.eval()
    network.eval()
    try:
        # fork_rng also spans VAE/pipeline construction: ``from_pretrained``
        # initializes weights and can consume the global torch RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            vae = load_vae(config)
            pipe = MOSSImagePipeline(dit, create_sample_scheduler(sample), vae=vae).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            negative = conditioning[sample["negative_prompt"]].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            negative_mask = torch.ones(negative.shape[:2], device="cuda", dtype=torch.bool)
            for index, prompt in enumerate(prompts):
                prompt_started = time.perf_counter()
                prompt_path = output_path if single_prompt else output_path.with_name(
                    f"{output_path.stem}_{index:03d}{output_path.suffix}"
                )
                positive = conditioning[prompt].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
                positive_mask = torch.ones(positive.shape[:2], device="cuda", dtype=torch.bool)
                latent = sample_latents(
                    pipe,
                    sample,
                    positive,
                    positive_mask,
                    negative,
                    negative_mask,
                    torch.Generator("cuda").manual_seed(sample["seed"]),
                    autocast_dtype=torch.bfloat16,
                )
                image = decode_latents(vae, latent, pipe.image_processor)[0]
                image.save(prompt_path)
                prompt_tokens = positive.shape[1]
                del latent, image, positive, positive_mask
                outputs.append({
                    "output_path": str(prompt_path),
                    "prompt_chars": len(prompt),
                    "prompt_tokens": prompt_tokens,
                    "negative_prompt_chars": len(sample["negative_prompt"]),
                    "negative_prompt_tokens": negative.shape[1],
                    "seed": sample["seed"],
                    "width": sample["width"],
                    "height": sample["height"],
                    "num_inference_steps": sample["num_inference_steps"],
                    "guidance_scale": sample["guidance_scale"],
                    "adapter_multiplier": network.multiplier,
                    "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "seconds": time.perf_counter() - prompt_started,
                })
            del negative, negative_mask
            allocated_gib = torch.cuda.memory_allocated() / 2**30
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        dit.train(dit_was_training)
        network.train(network_was_training)
        del pipe, vae
        gc.collect()
        torch.cuda.empty_cache()

    seconds = time.perf_counter() - started
    if single_prompt:
        metadata = outputs[0]
        metadata["allocated_gib"] = allocated_gib
        metadata["seconds"] = seconds
    else:
        metadata = {"outputs": outputs, "seconds": seconds}
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_paths = {name: str(path) for name, path in resolve_model_paths(config).items()}
    sample = config["sample"]
    single_prompt = isinstance(sample["prompt"], str)
    prompts = [sample["prompt"]] if single_prompt else sample["prompt"]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    captions = list(dict.fromkeys([*prompts, sample["negative_prompt"]]))
    encoding_started = time.perf_counter()
    conditioning = dict(encode_text(config, captions))
    text_encoding_seconds = time.perf_counter() - encoding_started
    logger.info("encoded %d unique texts: text_encoding_seconds=%.3f", len(conditioning), text_encoding_seconds)

    dit = load_dit(config)
    network = create_adapter(dit, config)
    weights = load_file(str(args.adapter))
    network.load_state_dict(weights, strict=True)
    network.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    print(f"Loaded {len(weights)} tensors into {len(network.loras)} DiT adapter modules", flush=True)
    adapter_sha256 = hashlib.sha256(args.adapter.read_bytes()).hexdigest()
    vae = load_vae(config)
    pipe = MOSSImagePipeline(dit, create_sample_scheduler(sample), vae=vae).to("cuda")

    comparisons = []
    with torch.inference_mode():
        negative = conditioning[sample["negative_prompt"]].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        negative_mask = torch.ones(negative.shape[:2], device="cuda", dtype=torch.bool)
        for index, prompt in enumerate(prompts):
            prompt_dir = args.output_dir if single_prompt else args.output_dir / f"prompt{index:03d}"
            if not single_prompt:
                prompt_dir.mkdir()
            positive = conditioning[prompt].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
            positive_mask = torch.ones(positive.shape[:2], device="cuda", dtype=torch.bool)
            images = {}
            latents = {}
            for name, multiplier in (("base", 0.0), ("lora", 1.0)):
                network.set_multiplier(multiplier)
                print(f"Sampling prompt {index} {name}, adapter_multiplier={multiplier}, seed={sample['seed']}", flush=True)
                latent = sample_latents(
                    pipe,
                    sample,
                    positive,
                    positive_mask,
                    negative,
                    negative_mask,
                    torch.Generator("cuda").manual_seed(sample["seed"]),
                )
                latents[name] = latent.cpu()
                image = decode_latents(vae, latent, pipe.image_processor)[0]
                image.save(prompt_dir / f"{name}.png")
                images[name] = np.asarray(image).astype(np.int16)
                del latent, image

            delta = np.abs(images["lora"] - images["base"])
            report = {
                "model_paths": model_paths,
                "adapter": str(args.adapter.resolve()),
                "adapter_sha256": adapter_sha256,
                "loaded_tensor_count": len(weights),
                "adapter_modules": len(network.loras),
                "sample": {
                    **{key: value for key, value in sample.items() if key not in ("prompt", "negative_prompt")},
                    "scheduler": {
                        key: pipe.scheduler.config[key]
                        for key in ("num_train_timesteps", "shift", "training_shift")
                    },
                    "prompt_chars": len(prompt),
                    "prompt_tokens": positive.shape[1],
                    "negative_prompt_chars": len(sample["negative_prompt"]),
                    "negative_prompt_tokens": negative.shape[1],
                },
                "text_encoding_seconds": text_encoding_seconds,
                "base_multiplier": 0.0,
                "lora_multiplier": 1.0,
                "latent_mean_abs_difference": (latents["lora"] - latents["base"]).abs().mean().item(),
                "latent_max_abs_difference": (latents["lora"] - latents["base"]).abs().max().item(),
                "image_mean_abs_difference_255": float(delta.mean()),
                "changed_pixel_fraction": float(np.any(delta != 0, axis=-1).mean()),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with (prompt_dir / "comparison.json").open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)
            comparisons.append({"prompt_index": index, "output_dir": str(prompt_dir), **report})
            print(
                f"Sample comparison saved: prompt_index={index} output={prompt_dir} "
                f"chars={len(prompt)} tokens={positive.shape[1]} "
                f"image_mae={report['image_mean_abs_difference_255']:.4f}",
                flush=True,
            )
            del positive, positive_mask, images, latents, delta
        del negative, negative_mask

    if not single_prompt:
        with (args.output_dir / "comparisons.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {"text_encoding_seconds": text_encoding_seconds, "comparisons": comparisons},
                handle,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
