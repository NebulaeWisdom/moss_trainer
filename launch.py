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
