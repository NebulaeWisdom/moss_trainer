"""Configuration, scoped restoration, and optional real-CUDA numerical contracts."""

import importlib.util
import os
import unittest

from moss_trainer.launch import configure_environment

configure_environment()  # Before torch/Triton imports: compiler artifacts stay in temp/.

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from moss_trainer import runtime
from moss_trainer.vendor import pipeline_moss_image as moss


def sdpa_flags():
    return tuple(getter() for getter in (
        torch.backends.cuda.math_sdp_enabled,
        torch.backends.cuda.flash_sdp_enabled,
        torch.backends.cuda.mem_efficient_sdp_enabled,
        torch.backends.cuda.cudnn_sdp_enabled,
    ))


def originals():
    return moss._situ, moss._rotate, moss._RMSNorm.forward


def set_cc(value):
    if value is None:
        os.environ.pop("CC", None)
    else:
        os.environ["CC"] = value


class ScopeAssertions:
    def check_scope(self, compile_pointwise, cc):
        saved_functions, saved_cc, saved_flags = originals(), os.environ.get("CC"), sdpa_flags()
        try:
            set_cc(cc)
            with sdpa_kernel(SDPBackend.MATH):
                self.assertEqual(sdpa_flags(), (True, False, False, False))
                for backend in ("auto", "cudnn"):
                    for fail in (False, True):
                        with self.subTest(backend=backend, fail=fail, cc=cc):
                            config = {"training": {"attention_backend": backend,
                                                   "compile_pointwise": compile_pointwise}}
                            try:
                                with runtime.training_kernels(config):
                                    expected = (True,) * 4 if backend == "cudnn" else (True, False, False, False)
                                    self.assertEqual(sdpa_flags(), expected)
                                    if fail:
                                        raise ArithmeticError("scope sentinel")
                            except ArithmeticError as error:
                                self.assertTrue(fail)
                                self.assertEqual(str(error), "scope sentinel")
                            else:
                                self.assertFalse(fail, "training context suppressed the sentinel exception")
                            self.assertEqual(sdpa_flags(), (True, False, False, False))
                            for actual, saved in zip(originals(), saved_functions):
                                self.assertIs(actual, saved)
                            self.assertEqual(os.environ.get("CC"), cc)
            self.assertEqual(sdpa_flags(), saved_flags)
        finally:
            set_cc(saved_cc)


class ConfigurationTests(ScopeAssertions, unittest.TestCase):
    def test_attention_defaults_and_explicit_choices(self):
        for resumed in (False, True):
            for supplied in (None, "auto", "cudnn"):
                with self.subTest(resumed=resumed, supplied=supplied):
                    config = {"training": {}}
                    if supplied is not None:
                        config["training"]["attention_backend"] = supplied
                    expected = supplied or ("auto" if resumed else "cudnn")
                    self.assertEqual(runtime.resolve_attention_backend(config, resumed=resumed), expected)
                    self.assertEqual(config["training"]["attention_backend"], expected)

    def test_unknown_attention_rejected_without_model_configuration(self):
        for invalid in ("flash", "CUDNN", "", None, False, 1):
            with self.subTest(invalid=invalid):
                config = {"training": {"attention_backend": invalid}}
                for resumed in (False, True):
                    with self.assertRaisesRegex(ValueError, "training.attention_backend"):
                        runtime.resolve_attention_backend(config, resumed=resumed)
                with self.assertRaisesRegex(ValueError, "training.attention_backend"):
                    with runtime.training_kernels(config):
                        self.fail("invalid attention setting entered the training body")

    def test_compile_defaults_false_and_rejects_non_booleans_before_loading(self):
        for training in ({}, {"compile_pointwise": False}):
            config = {"training": training}
            self.assertIs(runtime.resolve_compile_pointwise(config), False)
            self.assertIs(training["compile_pointwise"], False)
        for invalid in (0, 1, "true", "false", None, [], {}):
            with self.subTest(invalid=invalid):
                config = {"training": {"attention_backend": "auto", "compile_pointwise": invalid}}
                with self.assertRaisesRegex(ValueError, "training.compile_pointwise must be a boolean"):
                    runtime.resolve_compile_pointwise(config)
                with self.assertRaisesRegex(ValueError, "training.compile_pointwise must be a boolean"):
                    with runtime.training_kernels(config):
                        self.fail("non-boolean compiler setting entered the training body")

    def test_attention_scope_restores_normal_and_exception_exits(self):
        self.check_scope(False, os.environ.get("CC"))


class CudaPointwiseTests(ScopeAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("real pointwise compilation requires CUDA")
        if importlib.util.find_spec("triton") is None:
            raise unittest.SkipTest("real pointwise compilation requires Triton")
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("real pointwise comparison requires CUDA BF16 support")
        # Installed-but-broken compilation is a failure, never a skip or eager fallback.
        config = {"training": {"compile_pointwise": True}}
        if runtime.resolve_compile_pointwise(config) is not True:
            raise AssertionError("explicit compile_pointwise=True was not preserved")

    def test_compiled_scope_restores_functions_flags_and_cc(self):
        for cc in (None, "", "kernel-contract-user-compiler"):
            self.check_scope(True, cc)

    def assert_numerically_close(self, actual, expected, *, gradient=False):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all().item(), "nonfinite compiled result")
        self.assertTrue(torch.isfinite(expected).all().item(), "nonfinite eager result")
        difference = actual.float() - expected.float()
        relative = difference.norm().item() / max(expected.float().norm().item(), 1e-30)
        maximum = difference.abs().max().item()
        absolute_limit = 1e-5 if gradient else 1e-6
        self.assertTrue(relative <= 0.01 or maximum <= absolute_limit,
                        f"relative L2={relative:.6g}, max absolute={maximum:.6g}")

    def compare_checkpointed(self, eager, compiled, values, cotangent):
        def evaluate(function):
            arguments = tuple(value.detach().clone().requires_grad_(True) for value in values)
            result = checkpoint(function, *arguments, use_reentrant=False)
            gradients = torch.autograd.grad(result, arguments, grad_outputs=cotangent)
            return result.detach(), gradients

        expected, expected_gradients = evaluate(eager)
        actual, actual_gradients = evaluate(compiled)
        self.assert_numerically_close(actual, expected)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            self.assert_numerically_close(actual_gradient, expected_gradient, gradient=True)

    def test_bf16_forward_and_checkpointed_input_gradients(self):
        eager_situ, eager_rotate, eager_norm = originals()
        generator = torch.Generator(device="cuda").manual_seed(173)

        def random(shape):
            return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)

        norms = {width: moss._RMSNorm(width).to(device="cuda", dtype=torch.bfloat16).requires_grad_(False)
                 for width in (128, 3072)}
        with torch.no_grad():
            for norm in norms.values():
                norm.weight.copy_(1 + 0.1 * random(norm.weight.shape))
        config = {"training": {"attention_backend": "auto", "compile_pointwise": True}}
        with runtime.training_kernels(config):
            for sequence in (31, 67):
                with self.subTest(operation="situ", sequence=sequence):
                    shape = (1, sequence, 3072)
                    self.compare_checkpointed(eager_situ, moss._situ,
                                              (3 * random(shape), 2 * random(shape)), random(shape))
                for rank in (3, 4):
                    with self.subTest(operation="rotate", sequence=sequence, rank=rank):
                        shape = (sequence, 24, 128) if rank == 3 else (1, sequence, 24, 128)
                        phase = random((*shape[:-2], 64)).float()
                        rope = torch.cat((phase.sin(), phase.cos()), -1).to(torch.bfloat16)
                        self.compare_checkpointed(eager_rotate, moss._rotate,
                                                  (random(shape), rope), random(shape))
                    with self.subTest(operation="rms_norm", sequence=sequence, rank=rank):
                        shape = (1, sequence, 3072) if rank == 3 else (1, sequence, 24, 128)
                        value, norm = random(shape), norms[shape[-1]]
                        eager = lambda x: eager_norm(norm, x)
                        # Frozen-input and input-grad paths, then nonreentrant recomputation.
                        self.assert_numerically_close(norm(value), eager(value))
                        self.compare_checkpointed(eager, norm, (value,), random(shape))
                        self.assertFalse(norm.weight.requires_grad)
                        self.assertIsNone(norm.weight.grad)


if __name__ == "__main__":
    unittest.main()
