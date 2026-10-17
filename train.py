"""Single-GPU MOSS adapter training with exact optimizer-boundary resume.

Frozen DiT BF16 forwards consume cached FP32 raw latents and frozen text
embeddings. Live previews temporarily load the frozen FP32 VAE. Trainable
adapter weights and optimizer state stay FP32; exported adapters are BF16.

    python -m moss_trainer.train --config config.json --cache-dir cache --output-dir NEW
    python -m moss_trainer.train --resume run/state.pt --output-dir NEW

Every run uses a new directory. State checkpoints retain the original config,
cache, scheduler horizon, RNG and next unread data cursor. Epochs and batches
are zero-based; an epoch-end cursor is the next epoch with next_batch=0.

Flow: x_sigma = (1 - sigma) * x0 + sigma * noise; timestep = 1000 * sigma;
target = noise - x0. An epoch's last accumulation window is never joined to
the next epoch and is normalized by its actual number of samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F
from tqdm import tqdm

from . import data, optim, runtime, sample

LATENT_STATS_NAME = "latent_stats.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="JSON config for a new training job")
    source.add_argument("--resume", type=Path, help="full-state checkpoint from a trusted run")
    parser.add_argument("--cache-dir", type=Path, help="runtime cache directory for --config")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if args.config is not None and args.cache_dir is None:
        parser.error("--cache-dir is required with --config")
    return args


def sample_sigma(count, sampling, logit_mean, logit_std, generator, device) -> torch.Tensor:
    """Per-sample flow sigma in (0, 1)."""
    if sampling == "uniform":
        return torch.rand(count, generator=generator, device=device, dtype=torch.float32)
    if sampling == "logit_normal":
        normal = torch.randn(count, generator=generator, device=device, dtype=torch.float32)
        return torch.sigmoid(normal * logit_std + logit_mean)
    raise ValueError(f"unsupported timestep_sampling: {sampling!r}")


def _sample_weighted_loss(loss: torch.Tensor, batch_size: int, window_samples: int) -> torch.Tensor:
    """Give each sample equal weight across an accumulation window's bucket batches."""
    return loss * batch_size / window_samples


def log_entry(handle, entry: dict) -> None:
    line = json.dumps(entry, ensure_ascii=False)
    handle.write(line + "\n")
    handle.flush()


def adapter_metadata(config: dict, global_step: int) -> dict[str, str]:
    adapter = config["adapter"]
    training = config["training"]
    return {
        "ss_network_module": f"lycoris.{adapter['algo']}",
        "ss_network_dim": str(adapter["rank"]),
        "ss_network_alpha": str(adapter["alpha"]),
        "ss_network_args": json.dumps(
            {
                "algo": adapter["algo"],
                "dropout": adapter["dropout"],
                "target_modules": adapter["target_modules"],
                **adapter["kwargs"],
            }
        ),
        "ss_base_model_version": "moss_image",
        "ss_model_type": "MOSSImage",
        "ss_arch": "MOSSImageTransformer",
        "ss_base_model": str(config["model_dir"]),
        "ss_text_encoder": "frozen",
        "ss_training_steps": str(global_step),
        "ss_learning_rate": str(config["optimizer"]["learning_rate"]),
        "ss_optimizer": config["optimizer"]["type"],
        "ss_timestep_sampling": str(training["timestep_sampling"]),
        "ss_seed": str(training["seed"]),
    }


def save_checkpoint(
    network, optimizers: dict, schedulers: dict, generator: torch.Generator,
    config: dict, cache_dir: Path, output_dir: Path, global_step: int,
    epoch: int, next_batch: int, *, final: bool = False,
) -> dict:
    """Save BF16 export and optional FP32 full state at an optimizer boundary."""
    stem = "adapter" if final else f"step{global_step:06d}"
    adapter_path = output_dir / f"{stem}.safetensors"
    network.save_weights(
        str(adapter_path), torch.bfloat16, adapter_metadata(config, global_step)
    )
    result = {"adapter": adapter_path.name, "state": None}
    if config["save"]["save_state"]:
        state_path = output_dir / ("state.pt" if final else f"{stem}.state.pt")
        state = {
            # LyCORIS state_dict is an export transform (e.g. folds scalar into w1).
            # Raw traversal also includes nonpersistent buffers; shared tensors use
            # the same canonical names as the freshly constructed resume network.
            "adapter": {
                name: tensor.detach().to(
                    device="cpu",
                    dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype,
                    copy=True,
                )
                for name, tensor in chain(network.named_parameters(), network.named_buffers())
            },
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "global_step": global_step,
            "epoch": epoch,
            "next_batch": next_batch,
            "rng": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
                "noise_generator": generator.get_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "config": config,
            "cache_dir": str(cache_dir),
        }
        with state_path.open("xb") as handle:
            torch.save(state, handle)
        result["state"] = state_path.name
    return result


def cadence_due(settings: dict, global_step: int, epoch: int, epoch_finished: bool) -> bool:
    """Step and completed-epoch triggers coalesce at a single optimizer boundary."""
    every_steps = int(settings["every_steps"])
    every_epochs = int(settings["every_epochs"])
    return bool(
        (every_steps and global_step % every_steps == 0)
        or (epoch_finished and every_epochs and epoch % every_epochs == 0)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    resume = None
    if args.resume is not None:
        # Full optimizer/Python/numpy state is pickle: only load trusted local checkpoints.
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = resume["config"]
        cache_dir = Path(resume["cache_dir"])
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        cache_dir = args.cache_dir.resolve()
    attention_backend = runtime.resolve_attention_backend(config, resumed=resume is not None)
    compile_pointwise = runtime.resolve_compile_pointwise(config)
    checkpoint_group_size = runtime.resolve_checkpoint_group_size(config)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    training = config["training"]
    preview_settings = config.get("sample", {}).get("preview", {})
    preview_enabled = preview_settings.get("enabled", False) and (
        int(preview_settings.get("every_steps", 0)) > 0
        or int(preview_settings.get("every_epochs", 0)) > 0
    )
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation"])
    seed = int(training["seed"])
    sampling = training["timestep_sampling"]
    logit_mean = float(training["logit_mean"])
    logit_std = float(training["logit_std"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    dataset = data.CachedDataset(cache_dir, config)
    sampler = data.BucketBatchSampler(dataset, batch_size, seed)
    loader = data.make_dataloader(dataset, sampler, config["loader"])
    steps_per_epoch = (sampler.total_batches + accumulation - 1) // accumulation
    if resume is None:
        caps = []
        if training["max_steps"] is not None:
            caps.append(int(training["max_steps"]))
        if training["epochs"] is not None:
            caps.append(int(training["epochs"]) * steps_per_epoch)
        config["scheduler"]["num_training_steps"] = min(caps)
    total_steps = int(config["scheduler"]["num_training_steps"])
    with (output_dir / "config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    preview_conditioning = None
    preview_text_encoding_seconds = 0.0
    if preview_enabled:
        sample_settings = config["sample"]
        prompts = sample_settings["prompt"]
        if isinstance(prompts, str):
            prompts = [prompts]
        captions = list(dict.fromkeys([*prompts, sample_settings["negative_prompt"]]))
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        encoding_started = time.perf_counter()
        try:
            # Exhaust the encoder before loading DiT; keep training RNG unchanged.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                preview_conditioning = dict(runtime.encode_text(config, captions))
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        preview_text_encoding_seconds = time.perf_counter() - encoding_started

    stats = load_file(str(cache_dir / LATENT_STATS_NAME), device="cuda")
    dit = runtime.load_dit(config, checkpointing=bool(training["gradient_checkpointing"]))
    network = runtime.create_adapter(dit, config)
    dit.train()
    network.train()
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    optimizers = optim.make_optimizers(network, config)
    schedulers = optim.make_schedulers(optimizers, config, total_steps)
    global_step = 0
    epoch = 0
    next_batch = 0
    if resume is not None:
        # Restore exact trainable parameterization without LyCORIS export-load hooks.
        with torch.no_grad():
            for name, tensor in chain(network.named_parameters(), network.named_buffers()):
                tensor.copy_(resume["adapter"][name])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(resume["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(resume["schedulers"][name])
        global_step = int(resume["global_step"])
        epoch = int(resume["epoch"])
        next_batch = int(resume["next_batch"])
        torch.set_rng_state(resume["rng"]["torch_cpu"])
        torch.cuda.set_rng_state_all(resume["rng"]["torch_cuda"])
        generator.set_state(resume["rng"]["noise_generator"])
        random.setstate(resume["rng"]["python"])
        np.random.set_state(resume["rng"]["numpy"])
        del resume
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    run_peak_vram_gib = 0.0
    batches = None
    tqdm.write(
        f"Training started: step={global_step}/{total_steps} output={output_dir} "
        f"checkpoint_group_size={checkpoint_group_size}",
        file=sys.stdout,
    )
    with (
        runtime.training_kernels(config),
        (output_dir / "train.jsonl").open("x", encoding="utf-8") as handle,
        tqdm(
            total=total_steps, initial=global_step, desc="Training", unit="step",
            mininterval=0.5, dynamic_ncols=True, file=sys.stdout,
        ) as progress,
    ):
        log_entry(
            handle,
            {
                "event": "start",
                "cache_dir": str(cache_dir),
                "output_dir": str(output_dir),
                "config": str(args.config) if args.config is not None else None,
                "resume": str(args.resume) if args.resume is not None else None,
                "restored_config": args.resume is not None,
                "argv": sys.argv,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "records": len(dataset.records),
                "virtual_records": len(dataset),
                "batches_per_epoch": sampler.total_batches,
                "steps_per_epoch": steps_per_epoch,
                "planned_steps": total_steps,
                "global_step": global_step,
                "epoch": epoch,
                "next_batch": next_batch,
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "loader": config["loader"],
                "attention_backend": attention_backend,
                "compile_pointwise": compile_pointwise,
                "checkpoint_group_size": checkpoint_group_size,
                "optimizers": {
                    name: {
                        "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
                        "parameters": sum(
                            parameter.numel()
                            for group in optimizer.param_groups for parameter in group["params"]
                        ),
                    }
                    for name, optimizer in optimizers.items()
                },
                "scheduler": config["scheduler"],
                "cursor_semantics": "zero-based next unread epoch/batch; epoch end becomes next epoch, batch 0",
                "preview_text_encoding_seconds": round(preview_text_encoding_seconds, 6),
                "preview_text_encoding_count": len(preview_conditioning) if preview_conditioning is not None else 0,
                "timing_scope": "host wall time; data wait is iterator setup/next only, excludes explicit GPU work; CUDA overlap is approximate; step excludes checkpoint/preview",
            },
        )
        while global_step < total_steps:
            step_started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            data_wait_seconds = 0.0
            if batches is None:
                sampler.set_epoch(epoch, start_batch=next_batch)
                wait_started = time.perf_counter()
                batches = iter(loader)
                data_wait_seconds += time.perf_counter() - wait_started

            step_epoch = epoch
            batch_in_epoch = next_batch
            microbatch_count = min(accumulation, sampler.total_batches - next_batch)
            window_samples = sampler.sample_count(next_batch, microbatch_count)
            used_lrs = {
                name: [float(group["lr"]) for group in optimizer.param_groups]
                for name, optimizer in optimizers.items()
            }
            total_loss = None
            buckets = []
            indices = []
            caption_stats = None
            batch_sizes = []
            for _ in range(microbatch_count):
                wait_started = time.perf_counter()
                batch = next(batches)
                data_wait_seconds += time.perf_counter() - wait_started
                raw = batch["raw_latents"].to(device="cuda", dtype=torch.float32, non_blocking=True)
                latents = runtime.normalize_latents(raw, stats)
                noise = torch.randn(latents.shape, generator=generator, device="cuda", dtype=torch.float32)
                sigma = sample_sigma(latents.shape[0], sampling, logit_mean, logit_std, generator, "cuda")
                noisy = (1.0 - sigma.view(-1, 1, 1, 1)) * latents + sigma.view(-1, 1, 1, 1) * noise
                timestep = 1000.0 * sigma
                context = batch["context"].to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
                mask = batch["attention_mask"].to(device="cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = dit(noisy, timestep, context, mask).sample
                actual_batch_size = len(batch["indices"])
                loss = _sample_weighted_loss(
                    F.mse_loss(prediction.float(), noise - latents), actual_batch_size, window_samples
                )
                loss.backward()
                if total_loss is None:
                    total_loss = loss.detach()
                else:
                    total_loss.add_(loss.detach())
                buckets.append(batch["bucket"])
                indices.append(batch["indices"])
                batch_stats = batch["caption_stats"]
                if caption_stats is None:
                    caption_stats = {**batch_stats, "levels": batch_stats["levels"].copy()}
                else:
                    caption_stats["count"] += batch_stats["count"]
                    caption_stats["dropped"] += batch_stats["dropped"]
                    for level in caption_stats["levels"]:
                        caption_stats["levels"][level] += batch_stats["levels"][level]
                    for key in ("chars_min", "tokens_min"):
                        caption_stats[key] = min(caption_stats[key], batch_stats[key])
                    for key in ("chars_max", "tokens_max"):
                        caption_stats[key] = max(caption_stats[key], batch_stats[key])
                batch_sizes.append(actual_batch_size)
                next_batch += 1
                # Release graph outputs and device inputs before a live-preview VAE is loaded.
                del prediction, loss, noisy, timestep, sigma, noise, latents, raw, context, mask, batch

            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["max_grad_norm"]))
            for optimizer in optimizers.values():
                optimizer.step()
            for scheduler in schedulers.values():
                scheduler.step()
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            epoch_finished = next_batch == sampler.total_batches
            if epoch_finished:
                epoch += 1
                next_batch = 0
                batches = None

            loss_value = total_loss.item()
            grad_norm_value = float(grad_norm)
            del total_loss, grad_norm
            peak_vram_gib = torch.cuda.max_memory_allocated() / 2**30
            run_peak_vram_gib = max(run_peak_vram_gib, peak_vram_gib)
            log_entry(
                handle,
                {
                    "event": "step",
                    "step": global_step,
                    "epoch": step_epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "next_epoch": epoch,
                    "next_batch": next_batch,
                    "completed_epochs": epoch,
                    "microbatch_count": microbatch_count,
                    "sample_count": window_samples,
                    "loss": loss_value,
                    "lr": used_lrs,
                    "grad_norm": grad_norm_value,
                    "data_wait_seconds": round(data_wait_seconds, 6),
                    "step_seconds": round(time.perf_counter() - step_started, 6),
                    "elapsed": round(time.perf_counter() - started, 3),
                    "data_indices": indices,
                    "caption_stats": caption_stats,
                    "buckets": buckets,
                    "batch_sizes": batch_sizes,
                    "allocated_vram_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "peak_vram_gib": round(peak_vram_gib, 3),
                    "reserved_vram_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
                    "peak_reserved_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                },
            )
            progress.set_postfix(
                {
                    "loss": f"{loss_value:.4f}",
                    "LR": "/".join(f"{lr:.2e}" for lrs in used_lrs.values() for lr in lrs),
                    "VRAM": f"{peak_vram_gib:.2f}G",
                    "tokens": f"{caption_stats['tokens_min']}-{caption_stats['tokens_max']}",
                },
                refresh=False,
            )
            progress.update(1)
            if cadence_due(config["save"], global_step, epoch, epoch_finished):
                paths = save_checkpoint(
                    network, optimizers, schedulers, generator, config, cache_dir,
                    output_dir, global_step, epoch, next_batch,
                )
                log_entry(handle, {"event": "checkpoint", "step": global_step, "epoch": epoch, "next_batch": next_batch, **paths})
                tqdm.write(
                    f"Checkpoint saved: step={global_step} adapter={paths['adapter']}",
                    file=sys.stdout,
                )
            if preview_enabled and cadence_due(preview_settings, global_step, epoch, epoch_finished):
                preview = sample.generate_preview(
                    dit, network, config, preview_conditioning, output_dir / f"preview{global_step:06d}.png"
                )
                log_entry(handle, {"event": "preview", "step": global_step, "epoch": epoch, "next_batch": next_batch, "preview": preview})
                tqdm.write(f"Preview saved: step={global_step}", file=sys.stdout)

        paths = save_checkpoint(
            network, optimizers, schedulers, generator, config, cache_dir,
            output_dir, global_step, epoch, next_batch, final=True,
        )
        log_entry(
            handle,
            {
                "event": "finish",
                "steps": global_step,
                "epoch": epoch,
                "next_batch": next_batch,
                "elapsed": round(time.perf_counter() - started, 3),
                "peak_vram_gib": round(run_peak_vram_gib, 3),
                **paths,
            },
        )
    tqdm.write(
        f"Training finished: steps={global_step} elapsed={time.perf_counter() - started:.1f}s "
        f"adapter={paths['adapter']}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
