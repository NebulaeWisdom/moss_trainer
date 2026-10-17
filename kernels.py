"""Scoped DiT pointwise compilation with eager per-operation precision."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def validate_pointwise_support() -> None:
    """Fail before model loading when the requested compiler is unavailable."""
    if not callable(getattr(torch, "compile", None)):
        raise RuntimeError("training.compile_pointwise=true requires torch.compile")

    # launch normally installs these before torch is imported; keep direct
    # trainer invocations from placing compiler caches outside the project.
    from .launch import CACHE_ENVIRONMENT, TEMP

    for variable in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        if not os.environ.get(variable):
            directory = TEMP / CACHE_ENVIRONMENT[variable]
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)

    try:
        importlib.import_module("triton")
        from torch import _inductor
    except ImportError as error:
        raise RuntimeError(
            "training.compile_pointwise=true requires usable Triton and torch Inductor installations"
        ) from error
    list_options = getattr(_inductor, "list_options", None)
    if not callable(list_options) or "emulate_precision_casts" not in list_options():
        raise RuntimeError(
            "training.compile_pointwise=true requires Inductor's emulate_precision_casts option; "
            "compiling without eager per-operation precision is not supported"
        )


def _compile(function):
    return torch.compile(
        function, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True}
    )


def _compile_rms_norm(rms_norm):
    # Dynamo caches by code object. Distinct definitions, not repeated calls to
    # one wrapper factory, separate rank/input-grad variants. Each remaining
    # view/stride and forward/recompute thread context keeps ordinary guards.
    def hidden_no_input_grad(x, weight):
        return rms_norm(x, weight)

    def hidden_input_grad(x, weight):
        return rms_norm(x, weight)

    def heads_no_input_grad(x, weight):
        return rms_norm(x, weight)

    def heads_input_grad(x, weight):
        return rms_norm(x, weight)

    hidden_no_input_grad, hidden_input_grad, heads_no_input_grad, heads_input_grad = (
        _compile(function)
        for function in (
            hidden_no_input_grad, hidden_input_grad, heads_no_input_grad, heads_input_grad
        )
    )

    def forward(self, x):
        # Metadata-only dispatch: no copies, detach, casts, bucket keys or
        # module objects inside the compiled graphs.
        if x.ndim == 4:
            if x.requires_grad:
                return heads_input_grad(x, self.weight)
            return heads_no_input_grad(x, self.weight)
        if x.requires_grad:
            return hidden_input_grad(x, self.weight)
        return hidden_no_input_grad(x, self.weight)

    return forward


@contextlib.contextmanager
def pointwise_kernels() -> Iterator[None]:
    """Compile only DiT pointwise operations; restore patches and owned CC."""
    from .vendor import pipeline_moss_image as moss

    original_situ = moss._situ
    original_rotate = moss._rotate
    original_norm = moss._RMSNorm.forward
    original_cc = os.environ.get("CC")
    owns_cc = False
    try:
        if os.name == "nt" and not original_cc:
            # A .pth can load Triton from a different venv than sysconfig's
            # platlib; locate its actual bundled TinyCC, before any kernel build.
            spec = importlib.util.find_spec("triton")
            if spec is not None and spec.origin:
                tcc = Path(spec.origin).resolve().parent / "runtime" / "tcc" / "tcc.exe"
                if tcc.is_file():
                    os.environ["CC"] = str(tcc)
                    owns_cc = True
                    logger.info("pointwise compiler CC=%s (Triton's bundled TinyCC)", tcc)

        moss._situ = _compile(original_situ)
        moss._rotate = _compile(original_rotate)
        moss._RMSNorm.forward = _compile_rms_norm(moss._rms_norm)
        logger.info(
            "enabled DiT pointwise compilation: dynamic=True fullgraph=True emulate_precision_casts=True"
        )
        yield
    finally:
        moss._situ = original_situ
        moss._rotate = original_rotate
        moss._RMSNorm.forward = original_norm
        if owns_cc:
            if original_cc is None:
                os.environ.pop("CC", None)
            else:
                os.environ["CC"] = original_cc
