"""Raw VAE latent caching and deterministic CPU loading for MOSS training.

``resize`` selects a nearest-aspect bucket near each resolution's area budget;
it may enlarge small images. ``no_upscale`` keeps native size below the cap,
otherwise shrinks proportionally, then crops both dimensions down to the
bucket step. Native mode does not apply the resize table's aspect-ratio clamp.
Identical native buckets for one image across resolution levels are cached once.

Bucket dimensions must be multiples of 16 for VAE compression plus DiT
patchification; use a positive ``bucket_step`` divisible by 16. Native images
must retain at least one bucket step on each side after the area cap.

Latents remain unnormalized posterior samples [32, H/8, W/8]. Only metadata
and a bounded text-embedding LRU stay in memory; virtual repeat indices and
explicit (index, epoch) requests make caption dropout independent of worker
count, prefetch, and resume position. Geometry/text changes need a new cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

MANIFEST_NAME = "manifest.json"


def _build_buckets(resolution: int, bucket_step: int, max_aspect_ratio: float) -> list[tuple[int, int]]:
    """Quantized nearest-aspect buckets with area approximately resolution squared."""
    area = resolution * resolution
    width_min = bucket_step * math.ceil(math.sqrt(area / max_aspect_ratio) / bucket_step)
    width_max = bucket_step * math.floor(math.sqrt(area * max_aspect_ratio) / bucket_step)
    buckets = []
    for width in range(max(bucket_step, width_min), width_max + 1, bucket_step):
        height = max(bucket_step, bucket_step * round(area / width / bucket_step))
        if width / height > max_aspect_ratio:
            continue
        buckets.append((width, height))
    return buckets


def _assign_bucket(aspect_ratio: float, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Nearest bucket by absolute aspect-ratio distance (kohya convention)."""
    return min(buckets, key=lambda bucket: abs(bucket[0] / bucket[1] - aspect_ratio))


def _crop_resize(
    image: Image.Image,
    bucket_width: int,
    bucket_height: int,
    random_crop: bool,
    rng: random.Random,
) -> Image.Image:
    """Crop ``image`` to the bucket aspect ratio, then resize to the bucket."""
    source_width, source_height = image.size
    target_aspect_ratio = bucket_width / bucket_height
    if source_width / source_height > target_aspect_ratio:
        crop_width = min(source_width, max(1, round(source_height * target_aspect_ratio)))
        crop_height = source_height
    else:
        crop_width = source_width
        crop_height = min(source_height, max(1, round(source_width / target_aspect_ratio)))

    if random_crop:
        left = rng.randint(0, source_width - crop_width) if crop_width < source_width else 0
        top = rng.randint(0, source_height - crop_height) if crop_height < source_height else 0
    else:
        left = (source_width - crop_width) // 2
        top = (source_height - crop_height) // 2

    if (crop_width, crop_height) != image.size:
        image = image.crop((left, top, left + crop_width, top + crop_height))
    if image.size == (bucket_width, bucket_height):
        return image
    downscale = crop_width > bucket_width or crop_height > bucket_height
    resample = Image.Resampling.LANCZOS if downscale else Image.Resampling.BICUBIC
    return image.resize((bucket_width, bucket_height), resample)


def _native_bucket(
    width: int, height: int, resolution: int, bucket_step: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the proportional downsize and its floor-quantized crop dimensions."""
    if width * height > resolution * resolution:
        scale = resolution / math.sqrt(width * height)
        resized = (math.floor(width * scale), math.floor(height * scale))
    else:
        resized = (width, height)
    bucket = tuple(side // bucket_step * bucket_step for side in resized)
    return resized, bucket


def _crop_native(
    image: Image.Image,
    resized: tuple[int, int],
    bucket: tuple[int, int],
    random_crop: bool,
    rng: random.Random,
) -> Image.Image:
    """Only downsize above the area cap, then trim quantization remainders."""
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.LANCZOS)
    width, height = bucket
    if image.size == bucket:
        return image
    if random_crop:
        left = rng.randint(0, image.width - width)
        top = rng.randint(0, image.height - height)
    else:
        left = (image.width - width) // 2
        top = (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def _stable_seed(*parts) -> int:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _to_pixel_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """``RGB`` image -> ``[1, 3, H, W]`` in ``[-1, 1]`` on ``device``/``dtype``."""
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.reshape(image.height, image.width, 3).permute(2, 0, 1)
    data = data.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)
    return data.unsqueeze(0).to(dtype)


def _caption_for(
    path: Path, default_caption: str | None, caption_extension: str, caption_format: str
) -> str | dict[str, str]:
    if caption_format == "moss_official":
        captions = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["caption"]
        return {level: captions[level] for level in ("long", "medium", "short")}
    caption_file = path.with_suffix(caption_extension)
    if caption_file.is_file():
        return caption_file.read_text(encoding="utf-8").strip()
    if default_caption is None:
        raise FileNotFoundError(f"missing caption file {caption_file}; dataset.caption is not set")
    return default_caption


def cache_dataset(
    data_config: dict,
    vae,
    cache_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> list[dict]:
    """Encode unique per-image bucket assignments into a new ``cache_dir``.

    ``vae`` must be a frozen ``AutoencoderKL`` (``AutoencoderKLFlux2``) already
    on ``device`` with its weights in ``dtype``: pixels are cast to ``dtype``
    and the cached latent keeps the VAE's native dtype. ``cache_dir`` must not
    already contain data (no silent overwrite of an existing cache).

    Returns the record list and writes ``cache_dir/manifest.json`` (config
    echo, records, bucket statistics).
    """
    image_dir = Path(data_config["image_dir"])
    caption_dir = data_config.get("caption_dir")
    caption_dir = Path(caption_dir) if caption_dir is not None else None
    caption_format = data_config.get("caption_format", "plain")
    default_caption = data_config.get("caption")
    caption_extension = data_config.get("caption_extension", ".txt")
    if caption_format == "plain" and not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"
    resolutions = [int(resolution) for resolution in data_config["resolutions"]]
    bucket_step = int(data_config.get("bucket_step", 64))
    bucket_mode = data_config.get("bucket_mode", "resize")
    max_aspect_ratio = float(data_config.get("max_aspect_ratio", 2.0))
    random_crop = bool(data_config.get("random_crop", False))
    seed = int(data_config.get("seed", 42))

    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"cache directory is not empty, refusing to overwrite: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"no images with {IMAGE_EXTENSIONS} under {image_dir}")

    bucket_tables = (
        {resolution: _build_buckets(resolution, bucket_step, max_aspect_ratio) for resolution in resolutions}
        if bucket_mode == "resize"
        else {}
    )
    logger.info(
        "caching %d images at %d resolutions, mode=%s, sequential PIL/VAE, into %s",
        len(image_paths),
        len(resolutions),
        bucket_mode,
        cache_dir,
    )

    vae.eval()
    vae.requires_grad_(False)

    records: list[dict] = []
    bucket_counts: dict[tuple[int, int, int], int] = {}
    duplicate_assignments = 0
    preprocessing_seconds = 0.0
    started = time.perf_counter()

    for image_path in tqdm(
        image_paths, file=sys.stdout, unit="img", desc="Cache images", dynamic_ncols=True, mininterval=0.5
    ):
        caption_path = image_path if caption_dir is None else caption_dir / image_path.relative_to(image_dir)
        caption = _caption_for(caption_path, default_caption, caption_extension, caption_format)
        assigned_buckets = set()
        preprocess_started = time.perf_counter()
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            preprocessing_seconds += time.perf_counter() - preprocess_started
            source_width, source_height = image.size
            for resolution in resolutions:
                if bucket_mode == "no_upscale":
                    resized, bucket = _native_bucket(source_width, source_height, resolution, bucket_step)
                    if not all(bucket):
                        raise ValueError(
                            f"no_upscale cannot fit {image_path} ({source_width}x{source_height}) "
                            f"at {resolution}px with bucket_step={bucket_step}"
                        )
                    if bucket in assigned_buckets:
                        duplicate_assignments += 1
                        continue
                else:
                    bucket = _assign_bucket(source_width / source_height, bucket_tables[resolution])
                assigned_buckets.add(bucket)
                bucket_width, bucket_height = bucket
                record_index = len(records)
                record_seed = _stable_seed(
                    seed, image_path.relative_to(image_dir).as_posix(), resolution, bucket
                )
                preprocess_started = time.perf_counter()
                if bucket_mode == "no_upscale":
                    pixels = _crop_native(image, resized, bucket, random_crop, random.Random(record_seed))
                else:
                    pixels = _crop_resize(
                        image, bucket_width, bucket_height, random_crop, random.Random(record_seed)
                    )
                preprocessing_seconds += time.perf_counter() - preprocess_started
                pixel_tensor = _to_pixel_tensor(pixels, device, dtype)
                generator = torch.Generator(device=device).manual_seed(record_seed)
                with torch.no_grad():
                    posterior = vae.encode(pixel_tensor).latent_dist
                    latent = posterior.sample(generator=generator)[0].to("cpu").contiguous()
                latent_path = cache_dir / f"{record_index:06d}.safetensors"
                save_file({"latent": latent}, str(latent_path))
                records.append(
                    {
                        "latent": str(latent_path.resolve()),
                        "caption": caption,
                        "width": bucket_width,
                        "height": bucket_height,
                        "source": str(image_path),
                    }
                )
                key = (resolution, bucket_width, bucket_height)
                bucket_counts[key] = bucket_counts.get(key, 0) + 1
                del pixels, pixel_tensor, posterior, latent

    bucket_statistics = {
        "total_records": len(records),
        "total_buckets": len(bucket_counts),
        "distinct_sizes": len({(width, height) for _, width, height in bucket_counts}),
        "duplicate_assignments_omitted": duplicate_assignments,
        "buckets": [
            {"resolution": resolution, "width": width, "height": height, "count": count}
            for (resolution, width, height), count in sorted(bucket_counts.items())
        ],
        "per_resolution": [
            {
                "resolution": resolution,
                "num_buckets": sum(1 for key in bucket_counts if key[0] == resolution),
                "num_records": sum(count for key, count in bucket_counts.items() if key[0] == resolution),
            }
            for resolution in resolutions
        ],
    }

    manifest = {
        "data_config": {
            "image_dir": str(image_dir),
            "caption_dir": str(caption_dir) if caption_dir is not None else None,
            "caption_format": caption_format,
            "caption": default_caption,
            "caption_extension": caption_extension,
            "resolutions": resolutions,
            "bucket_step": bucket_step,
            "bucket_mode": bucket_mode,
            "max_aspect_ratio": max_aspect_ratio,
            "random_crop": random_crop,
            "seed": seed,
        },
        "num_images": len(image_paths),
        "num_records": len(records),
        "bucket_statistics": bucket_statistics,
        "records": records,
    }
    with (cache_dir / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    logger.info(
        "cached %d images into %d records in %d resolution buckets (%d distinct sizes), "
        "%d duplicates omitted, %.2fs, %.2f images/s, %.2f records/s overall; "
        "PIL preprocessing=%.2fs (decode + crop/resize only); manifest: %s",
        len(image_paths),
        len(records),
        bucket_statistics["total_buckets"],
        bucket_statistics["distinct_sizes"],
        duplicate_assignments,
        elapsed,
        len(image_paths) / elapsed,
        len(records) / elapsed,
        preprocessing_seconds,
        cache_dir / MANIFEST_NAME,
    )
    return records


class CachedDataset(Dataset):
    """Lazy CPU tensors, metadata read once, and virtual repeats of cached records."""

    def __init__(self, cache_dir: Path, config: dict):
        cache_dir = Path(cache_dir)
        with (cache_dir / MANIFEST_NAME).open(encoding="utf-8") as handle:
            self.records = json.load(handle)["records"]
        with (cache_dir / "text_manifest.json").open(encoding="utf-8") as handle:
            self.text_manifest = json.load(handle)
        data_config = config["dataset"]
        self.seed = int(data_config.get("seed", config["training"].get("seed", 42)))
        self.repeats = int(data_config.get("repeats", 1))
        self.caption_dropout = float(data_config.get("caption_dropout", 0.0))
        self.caption_format = data_config.get("caption_format", "plain")
        if self.caption_format == "moss_official":
            weights = data_config.get("caption_weights", {})
            caption_weights = tuple(float(weights.get(level, 1.0)) for level in ("long", "medium", "short"))
            total_weight = sum(caption_weights)
            self.caption_weights = tuple(weight / total_weight for weight in caption_weights)
        self.bucket_keys = [(record["width"], record["height"]) for record in self.records] * self.repeats
        self._embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        logger.info(
            "loaded cache metadata: %d records, repeats=%d, %d virtual indices, "
            "%d bucket sizes, %d encoded captions, embedding LRU=32 per worker",
            len(self.records),
            self.repeats,
            len(self),
            len(set(self.bucket_keys)),
            len(self.text_manifest),
        )

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(self, item: tuple[int, int]) -> dict:
        index, epoch = item
        record = self.records[index % len(self.records)]
        rng = random.Random(_stable_seed(self.seed, epoch, index))
        caption = record["caption"]
        level = "plain"
        if self.caption_format == "moss_official":
            level = rng.choices(("long", "medium", "short"), weights=self.caption_weights)[0]
            caption = caption[level]
        caption_dropped = rng.random() < self.caption_dropout
        caption = "" if caption_dropped else caption
        embedding = self._embedding_cache.get(caption)
        if embedding is None:
            embedding = load_file(self.text_manifest[caption], device="cpu")["embedding"].to(torch.bfloat16)
            self._embedding_cache[caption] = embedding
            if len(self._embedding_cache) > 32:
                self._embedding_cache.popitem(last=False)
        else:
            self._embedding_cache.move_to_end(caption)
        return {
            "latent": load_file(record["latent"], device="cpu")["latent"],
            "embedding": embedding,
            "caption_level": level,
            "caption_chars": len(caption),
            "caption_tokens": int(embedding.shape[0]),
            "caption_dropped": caption_dropped,
            "width": record["width"],
            "height": record["height"],
            "index": index,
        }


class BucketBatchSampler(Sampler):
    """Homogeneous shuffled batches, including tails; resume without loading skipped data."""

    def __init__(self, dataset: CachedDataset, batch_size: int, seed: int):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        self._buckets: dict[tuple[int, int], list[int]] = {}
        for index, key in enumerate(dataset.bucket_keys):
            self._buckets.setdefault(key, []).append(index)
        self._bucket_order = sorted(self._buckets)
        self.total_batches = sum((len(indices) + batch_size - 1) // batch_size for indices in self._buckets.values())

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        return self.total_batches - self.start_batch

    def __iter__(self):
        epoch, start_batch = self.epoch, self.start_batch
        rng = random.Random(_stable_seed(self.seed, epoch, "batches"))
        batches = []
        for key in self._bucket_order:
            indices = self._buckets[key].copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
        rng.shuffle(batches)
        for position in range(start_batch, len(batches)):
            yield [(index, epoch) for index in batches[position]]


def collate_cached(items: list[dict]) -> dict:
    """Stack raw CPU latents and pad real text embeddings with a boolean token mask."""
    length = max(item["embedding"].shape[0] for item in items)
    context = torch.zeros(len(items), length, items[0]["embedding"].shape[-1], dtype=torch.bfloat16)
    attention_mask = torch.zeros(len(items), length, dtype=torch.bool)
    caption_stats = {
        "count": len(items),
        "levels": {"plain": 0, "long": 0, "medium": 0, "short": 0},
        "dropped": 0,
        "chars_min": items[0]["caption_chars"],
        "chars_max": items[0]["caption_chars"],
        "tokens_min": items[0]["caption_tokens"],
        "tokens_max": items[0]["caption_tokens"],
    }
    for index, item in enumerate(items):
        row = item["embedding"]
        context[index, : row.shape[0]].copy_(row)
        attention_mask[index, : row.shape[0]] = True
        caption_stats["levels"][item["caption_level"]] += 1
        caption_stats["dropped"] += int(item["caption_dropped"])
        caption_stats["chars_min"] = min(caption_stats["chars_min"], item["caption_chars"])
        caption_stats["chars_max"] = max(caption_stats["chars_max"], item["caption_chars"])
        caption_stats["tokens_min"] = min(caption_stats["tokens_min"], item["caption_tokens"])
        caption_stats["tokens_max"] = max(caption_stats["tokens_max"], item["caption_tokens"])
    return {
        "raw_latents": torch.stack([item["latent"] for item in items]),
        "context": context,
        "attention_mask": attention_mask,
        "caption_stats": caption_stats,
        "bucket": [items[0]["width"], items[0]["height"]],
        "indices": [item["index"] for item in items],
    }


def make_dataloader(dataset: CachedDataset, sampler: BucketBatchSampler, loader_config: dict) -> DataLoader:
    """Pin CPU batches after collation; worker startup never consumes training RNG."""
    num_workers = int(loader_config.get("num_workers", 2))
    worker_options = {}
    if num_workers:
        worker_options["prefetch_factor"] = int(loader_config.get("prefetch_factor", 2))
        worker_options["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_cached,
        num_workers=num_workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        generator=torch.Generator(device="cpu").manual_seed(sampler.seed),
        **worker_options,
    )
