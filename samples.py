"""Sample a shared-model prompt/seed/multiplier matrix; omit --adapter for --multiplier 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "moss_trainer"

from .launch import configure_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path, nargs="+",
        help="Adapter checkpoint paths; optional only when all multipliers are zero.",
    )
    parser.add_argument("--multiplier", type=float, nargs="+", default=[1.0])
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompts", nargs="+")
    prompt_source.add_argument("--prompts-file", type=Path)
    parser.add_argument("--seed", type=int, nargs="+")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample = config["sample"].copy()
    if args.prompts_file is not None:
        prompts = [
            line for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.prompts is not None:
        prompts = args.prompts
    else:
        prompts = [sample["prompt"]] if isinstance(sample["prompt"], str) else sample["prompt"]
    if not prompts:
        parser.error("At least one prompt is required.")
    negative_prompt = sample["negative_prompt"] if args.negative_prompt is None else args.negative_prompt
    for argument, key in (
        ("width", "width"),
        ("height", "height"),
        ("steps", "num_inference_steps"),
        ("cfg", "guidance_scale"),
    ):
        value = getattr(args, argument)
        if value is not None:
            sample[key] = value
    seeds = list(dict.fromkeys(args.seed if args.seed is not None else [sample["seed"]]))
    multipliers = list(dict.fromkeys(args.multiplier))
    nonzero_multipliers = [multiplier for multiplier in multipliers if multiplier != 0.0]
    include_base = 0.0 in multipliers
    if args.adapter is None and nonzero_multipliers:
        parser.error("--adapter is required unless all --multiplier values are zero.")
    adapter_paths = [path.resolve() for path in (args.adapter or [])]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    configure_environment()
    # Suppress loader/encoder bars before importing their tqdm instances. Only
    # the image-level bar below explicitly opts back in.
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    import torch
    from safetensors.torch import load_file
    from tqdm import tqdm

    from .runtime import create_adapter, encode_text, load_dit, load_vae
    from .sample import decode_latents, sample_latents
    from .vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler

    phase_timings = {}
    load_counts = {"text_encoder": 0, "dit": 0, "vae": 0, "adapter_network": 0, "adapter_checkpoints": 0}
    captions = list(dict.fromkeys([*prompts, negative_prompt]))
    phase_started = time.perf_counter()
    print(f"Loading text encoder for {len(captions)} unique texts", flush=True)
    # Exhaustion runs encode_text's finally block, releasing the TE before DiT.
    conditioning = dict(encode_text(config, captions))
    phase_timings["text_encoding_seconds"] = time.perf_counter() - phase_started
    load_counts["text_encoder"] += 1
    print(f"Encoded {len(conditioning)} texts; text encoder released", flush=True)

    phase_started = time.perf_counter()
    print("Loading DiT", flush=True)
    dit = load_dit(config)
    phase_timings["dit_load_seconds"] = time.perf_counter() - phase_started
    load_counts["dit"] += 1

    network = None
    phase_timings["adapter_create_seconds"] = 0.0
    if adapter_paths:
        phase_started = time.perf_counter()
        network = create_adapter(dit, config)
        network.to(dtype=torch.bfloat16).eval().requires_grad_(False)
        phase_timings["adapter_create_seconds"] = time.perf_counter() - phase_started
        load_counts["adapter_network"] += 1

    phase_started = time.perf_counter()
    print("Loading VAE", flush=True)
    vae = load_vae(config)
    phase_timings["vae_load_seconds"] = time.perf_counter() - phase_started
    load_counts["vae"] += 1

    phase_started = time.perf_counter()
    pipe = MOSSImagePipeline(dit, MOSSImageScheduler(), vae=vae).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    phase_timings["pipeline_setup_seconds"] = time.perf_counter() - phase_started
    phase_timings["adapter_load_seconds"] = 0.0
    phase_timings["adapter_hash_seconds"] = 0.0
    phase_timings["sampling_seconds"] = 0.0
    adapters = []
    images = []
    image_count = len(prompts) * len(seeds) * (
        int(include_base) + len(adapter_paths) * len(nonzero_multipliers)
    )
    generator = torch.Generator("cuda")

    with torch.inference_mode(), tqdm(
        total=image_count, desc="Images", unit="image", file=sys.stdout,
        dynamic_ncols=True, mininterval=0.5, disable=False,
    ) as progress:
        negative = conditioning[negative_prompt].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
        negative_mask = torch.ones(negative.shape[:2], device="cuda", dtype=torch.bool)
        for adapter_index, adapter_path in enumerate(adapter_paths or [None]):
            if adapter_path is not None:
                phase_started = time.perf_counter()
                weights = load_file(str(adapter_path), device="cpu")
                network.load_state_dict(weights, strict=True)
                tensor_count = len(weights)
                del weights
                adapter_load_seconds = time.perf_counter() - phase_started
                phase_timings["adapter_load_seconds"] += adapter_load_seconds
                load_counts["adapter_checkpoints"] += 1

                phase_started = time.perf_counter()
                with adapter_path.open("rb") as handle:
                    adapter_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
                adapter_hash_seconds = time.perf_counter() - phase_started
                phase_timings["adapter_hash_seconds"] += adapter_hash_seconds
                adapters.append({
                    "index": adapter_index,
                    "path": str(adapter_path),
                    "sha256": adapter_sha256,
                    "tensor_count": tensor_count,
                    "load_seconds": adapter_load_seconds,
                    "hash_seconds": adapter_hash_seconds,
                })
                progress.write(
                    f"Loaded adapter{adapter_index:02d} {adapter_path.name}: {tensor_count} tensors",
                    file=sys.stdout,
                )
            active_multipliers = (
                [0.0] if include_base and adapter_index == 0 else []
            ) + nonzero_multipliers
            if not active_multipliers:
                continue
            for prompt_index, prompt in enumerate(prompts):
                positive = conditioning[prompt].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
                positive_mask = torch.ones(positive.shape[:2], device="cuda", dtype=torch.bool)
                for seed in seeds:
                    for multiplier in active_multipliers:
                        image_started = time.perf_counter()
                        if network is not None:
                            network.set_multiplier(multiplier)
                        latent = sample_latents(
                            pipe, sample, positive, positive_mask, negative, negative_mask,
                            generator.manual_seed(seed),
                        )
                        image = decode_latents(vae, latent, pipe.image_processor)[0]
                        is_base = multiplier == 0.0
                        suffix = "base" if is_base else (
                            f"adapter{adapter_index:02d}_{adapter_path.stem}_"
                            f"m{str(multiplier).removesuffix('.0')}"
                        )
                        image_path = output_dir / f"prompt{prompt_index:03d}_seed{seed}_{suffix}.png"
                        with image_path.open("xb") as handle:
                            image.save(handle, format="PNG")
                        del latent, image
                        seconds = time.perf_counter() - image_started
                        phase_timings["sampling_seconds"] += seconds
                        images.append({
                            "prompt_index": prompt_index,
                            "prompt_tokens": positive.shape[1],
                            "seed": seed,
                            "source_checkpoint": None if is_base else str(adapter_path),
                            "adapter_index": None if is_base else adapter_index,
                            "multiplier": multiplier,
                            "path": str(image_path),
                            "seconds": seconds,
                        })
                        progress.update(1)
                        progress.write(f"Saved {image_path.name}", file=sys.stdout)
                del positive, positive_mask
        negative_tokens = negative.shape[1]
        del negative, negative_mask

    phase_timings["total_seconds"] = time.perf_counter() - started
    parameters = {
        "config_source": str(args.config.resolve()),
        "model_dir": str(Path(config["model_dir"]).resolve()),
        "config": config,
        "inputs": {
            key: [str(item) if isinstance(item, Path) else item for item in value]
            if isinstance(value, list) else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "prompts": prompts,
        "negative_prompt": negative_prompt,
        "negative_prompt_tokens": negative_tokens,
        "seeds": seeds,
        "multipliers": multipliers,
        "sample": {
            key: sample[key]
            for key in ("width", "height", "num_inference_steps", "guidance_scale")
        },
        "adapters": adapters,
        "adapter_modules": len(network.loras) if network is not None else 0,
        "images": images,
        "phase_timings": phase_timings,
        "load_counts": load_counts,
    }
    with (output_dir / "parameters.json").open("x", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2, ensure_ascii=False)
    print(f"Saved parameters.json; {len(images)} images", flush=True)


if __name__ == "__main__":
    main()
