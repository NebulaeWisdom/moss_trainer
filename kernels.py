"""Scoped DiT pointwise and main-block compilation with eager precision casts."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import os
import time
from collections.abc import Iterator
from functools import wraps
from pathlib import Path
from types import MethodType

import torch

logger = logging.getLogger(__name__)

BLOCK_COMPILE_MODES = ("default", "max-autotune-no-cudagraphs")

_BLOCK_OPTIONS = {
    "emulate_precision_casts": True,
    "triton.cudagraphs": False,
    "autotune_cudagraph_benchmarking": False,
    "fallback_random": True,  # ATen RNG/dropout operators stay inside the full graph.
}
_BLOCK_DYNAMO_SETTINGS = {
    "suppress_errors": False,
    "fail_on_recompile_limit_hit": True,
    "skip_nnmodule_hook_guards": False,
    # Python adapter scales remain guarded constants, not FP64 CPU tensors.
    "specialize_float": True,
}


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


def validate_block_support(mode: str = "default") -> dict:
    """Validate support and return the mode's merged Inductor options."""
    if type(mode) is not str or mode not in BLOCK_COMPILE_MODES:
        raise ValueError(f"training.compile_blocks_mode must be one of {BLOCK_COMPILE_MODES}, got {mode!r}")
    validate_pointwise_support()
    from torch import _inductor
    from torch._dynamo import config

    if not callable(getattr(_inductor, "list_mode_options", None)):
        raise RuntimeError("training.compile_blocks requires torch._inductor.list_mode_options")
    options = dict(_inductor.list_mode_options(mode=mode, dynamic=True))
    options.update(_BLOCK_OPTIONS)
    if mode == "default":
        # Keep the default mode free of environment-enabled GEMM/max autotuning.
        options.update(max_autotune=False, max_autotune_gemm=False,
                       max_autotune_pointwise=False, coordinate_descent_tuning=False)
    missing = options.keys() - set(_inductor.list_options())
    if missing:
        raise RuntimeError(f"training.compile_blocks requires Inductor options: {sorted(missing)}")
    missing = [name for name in (*_BLOCK_DYNAMO_SETTINGS, "disable") if not hasattr(config, name)]
    if missing or not callable(getattr(config, "patch", None)):
        raise RuntimeError(f"training.compile_blocks requires scoped Dynamo settings: {missing}")
    if config.disable:
        raise RuntimeError("training.compile_blocks requires enabled Dynamo; eager fallback is not supported")
    return options


def _compile(function, *, options=None):
    return torch.compile(
        function, backend="inductor", dynamic=True, fullgraph=True,
        options={"emulate_precision_casts": True} if options is None else options,
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


def _compile_block_forward(original_forward, options):
    from torch._dynamo import config

    # Separate code objects share graphs across blocks without closing over a
    # particular layer. Normal guards retain adapter/module state correctness.
    def preview(self, x, rope, modulation):
        return original_forward(self, x, rope, modulation)

    def training_frozen_input(self, x, rope, modulation):
        return original_forward(self, x, rope, modulation)

    def training_input_grad(self, x, rope, modulation):
        return original_forward(self, x, rope, modulation)

    lanes = tuple(_compile(fn, options=options)
                  for fn in (preview, training_frozen_input, training_input_grad))
    seen = [False, False, False]

    def call_compiled(self, x, rope, modulation):
        lane = 0 if not torch.is_grad_enabled() else 2 if x.requires_grad else 1
        if seen[lane]:
            return lanes[lane](self, x, rope, modulation)
        started = time.perf_counter()
        output = lanes[lane](self, x, rope, modulation)
        seen[lane] = True
        logger.info(
            "block compiler first call: lane=%s shape=%s host_seconds=%.6f "
            "(includes lazy forward compilation, excludes lazy backward compilation)",
            ("preview", "training_frozen_input", "training_input_grad")[lane],
            tuple(x.shape), time.perf_counter() - started,
        )
        return output

    @wraps(original_forward)
    def forward(self, x, rope, modulation):
        # Config overrides are thread-local in supported torch builds. Checkpoint
        # recomputation can run on an autograd worker without the lifecycle patch.
        if (config.specialize_float and not config.suppress_errors
                and config.fail_on_recompile_limit_hit and not config.skip_nnmodule_hook_guards):
            return call_compiled(self, x, rope, modulation)
        with config.patch(_BLOCK_DYNAMO_SETTINGS):
            return call_compiled(self, x, rope, modulation)

    return forward


@contextlib.contextmanager
def block_kernels(dit, network, *, mode: str = "default") -> Iterator[None]:
    """Compile adapted main-block forwards inside the enclosing pointwise scope."""
    from lycoris.modules.locon import LoConModule
    from torch._dynamo import config
    from .vendor.pipeline_moss_image import _Block

    options = validate_block_support(mode)
    blocks = tuple(dit.layers)
    if not blocks or any(type(block) is not _Block for block in blocks):
        raise RuntimeError("training.compile_blocks requires the original main _Block classes")
    original_forward = _Block.forward
    if any(not isinstance(block.forward, MethodType)
           or block.forward.__self__ is not block
           or block.forward.__func__ is not original_forward for block in blocks):
        raise RuntimeError("training.compile_blocks requires unmodified main block forwards")
    if any(p.requires_grad for p in dit.parameters()):
        raise RuntimeError("training.compile_blocks requires a frozen base DiT")
    parameters = tuple(network.parameters())
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 for p in parameters):
        raise RuntimeError("training.compile_blocks requires trainable FP32 LyCORIS parameters")
    adapters = tuple(network.loras)
    if not adapters or any(type(adapter) is not LoConModule for adapter in adapters):
        raise RuntimeError("training.compile_blocks supports only adapter.algo='lora'")
    if any(not isinstance(adapter.org_module[0].forward, MethodType)
           or adapter.org_module[0].forward.__self__ is not adapter for adapter in adapters):
        raise RuntimeError("training.compile_blocks must be entered after LyCORIS apply_to")
    missing = object()
    saved = []
    with config.patch(_BLOCK_DYNAMO_SETTINGS):
        dispatch = _compile_block_forward(original_forward, options)
        try:
            for block in blocks:
                previous = block.__dict__.get("forward", missing)
                block.forward = MethodType(dispatch, block)
                saved.append((block, previous))
            logger.info(
                "enabled DiT block compilation: blocks=%d mode=%s dynamic=True fullgraph=True options=%s "
                "Dynamo settings=%s (also scoped per calling thread)",
                len(blocks), mode, options, _BLOCK_DYNAMO_SETTINGS,
            )
            yield
        finally:
            for block, previous in saved:
                if previous is missing:
                    del block.forward
                else:
                    block.forward = previous
