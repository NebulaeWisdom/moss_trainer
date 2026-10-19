"""Run a trainer stage with every runtime cache contained in the project.

The ``if __name__ == "__main__"`` guard is required: on Windows, DataLoader worker
processes are spawned by re-importing this file as ``__mp_main__``, so any top-level
stage dispatch would recursively launch the trainer instead of starting a worker.
The cache environment is inherited by those children through ``os.environ``.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMP = ROOT / "temp"

CACHE_ENVIRONMENT = {
    "TEMP": "process-temp",
    "TMP": "process-temp",
    "PIP_CACHE_DIR": "pip-cache",
    "HF_HOME": "hf-home",
    "TORCH_HOME": "torch-home",
    "TRITON_CACHE_DIR": "triton-cache",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor-cache",
    "CUDA_CACHE_PATH": "cuda-cache",
    "MPLCONFIGDIR": "matplotlib-cache",
}

STAGES = {"cache": "runtime", "train": "train", "sample": "sample"}


def resolve_model_paths(config: dict, *, base_dir: Path | None = None) -> dict[str, Path]:
    """Resolve component directories without importing model or UI dependencies."""
    base_dir = Path.cwd() if base_dir is None else Path(base_dir).resolve()
    root = config.get("model_dir")
    if root is not None and not isinstance(root, str):
        raise ValueError("model_dir must be a directory path string")
    root = root.strip() if root else ""
    model_dir = Path(root).expanduser() if root else None
    overrides = config.get("model_paths")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("model_paths must be an object")
    paths = {}
    for component in ("transformer", "vae", "text_encoder", "tokenizer"):
        override = overrides.get(component)
        if override is not None and not isinstance(override, str):
            raise ValueError(f"model_paths.{component} must be a directory path string or null")
        override = override.strip() if override else ""
        if override:
            path = Path(override).expanduser()
        elif model_dir is not None:
            path = model_dir / component
        else:
            raise ValueError(f"model_dir is required when model_paths.{component} is not set")
        paths[component] = (path if path.is_absolute() else base_dir / path).resolve()
    return paths


def configure_environment() -> None:
    """Redirect every toolchain cache into ``temp/`` and force offline model loading."""
    for variable, directory in CACHE_ENVIRONMENT.items():
        path = TEMP / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["LYCORIS_KERNEL_BACKEND"] = "torch"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))


def main() -> None:
    configure_environment()
    stage = STAGES[sys.argv.pop(1)]
    runpy.run_module(f"moss_trainer.{stage}", run_name="__main__")


if __name__ == "__main__":
    main()
