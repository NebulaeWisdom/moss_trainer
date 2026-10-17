"""Configuration, scoped restoration, and optional real-CUDA numerical contracts."""

import importlib.util
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from moss_trainer.launch import configure_environment

configure_environment()  # Before torch/Triton imports: compiler artifacts stay in temp/.

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from moss_trainer import kernels, runtime
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

    def test_block_compile_defaults_and_strict_dependencies(self):
        for training in ({}, {"compile_blocks": False}):
            config = {"training": training}
            self.assertIs(runtime.resolve_compile_blocks(config), False)
            self.assertIs(training["compile_blocks"], False)
            self.assertNotIn("compile_blocks_mode", training)
        for invalid in (0, 1, "true", "false", None, [], {}):
            config = {"training": {"attention_backend": "auto", "compile_blocks": invalid}}
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "compile_blocks must be a boolean"):
                    runtime.resolve_compile_blocks(config)
                with self.assertRaisesRegex(ValueError, "compile_blocks must be a boolean"):
                    with runtime.training_kernels(config):
                        self.fail("invalid block setting entered training")
        config = {"training": {"attention_backend": "auto", "compile_blocks": True},
                  "adapter": {"algo": "lora"}}
        with self.assertRaisesRegex(ValueError, "requires training.compile_pointwise=true"):
            runtime.resolve_compile_blocks(config)
        self.assertNotIn("compile_pointwise", config["training"])
        config["training"]["compile_pointwise"] = True
        for missing in ({}, {"dit": object()}, {"network": object()}):
            with self.assertRaisesRegex(ValueError, "requires the adapted dit and network"):
                with runtime.training_kernels(config, **missing):
                    self.fail("block scope entered without both models")
        for algo in ("lokr", "locon", None):
            config["adapter"]["algo"] = algo
            with self.assertRaisesRegex(ValueError, "supports only adapter.algo='lora'"):
                runtime.resolve_compile_blocks(config)
            with self.assertRaisesRegex(ValueError, "supports only adapter.algo='lora'"):
                with runtime.training_kernels(config, dit=object(), network=object()):
                    self.fail("unsupported adapter entered training")

    def test_block_mode_validation_and_persistence(self):
        for supplied in ({}, {"compile_blocks_mode": "default"},
                         {"compile_blocks_mode": "max-autotune-no-cudagraphs"}):
            config = {"training": {"compile_blocks": True, "compile_pointwise": True, **supplied},
                      "adapter": {"algo": "lora"}}
            with patch.object(kernels, "validate_block_support"):
                self.assertIs(runtime.resolve_compile_blocks(config), True)
                expected = supplied.get("compile_blocks_mode", "default")
                self.assertEqual(config["training"]["compile_blocks_mode"], expected)
                restored = {"training": dict(config["training"]), "adapter": dict(config["adapter"])}
                self.assertIs(runtime.resolve_compile_blocks(restored), True)
                self.assertEqual(restored["training"]["compile_blocks_mode"], expected)
        for invalid in (None, False, 1, [], {}, "", "max-autotune", "reduce-overhead"):
            config = {"training": {"attention_backend": "auto", "compile_blocks": True,
                                   "compile_pointwise": True, "compile_blocks_mode": invalid},
                      "adapter": {"algo": "lora"}}
            with self.subTest(mode=invalid):
                with self.assertRaisesRegex(ValueError, "compile_blocks_mode must be one of"):
                    runtime.resolve_compile_blocks(config)
                with self.assertRaisesRegex(ValueError, "compile_blocks_mode must be one of"):
                    with runtime.training_kernels(config, dit=object(), network=object()):
                        self.fail("unsupported compiler mode entered training")
            config["training"].update(compile_blocks=False, compile_pointwise=False)
            self.assertIs(runtime.resolve_compile_blocks(config), False)
            self.assertEqual(config["training"]["compile_blocks_mode"], invalid)
            with patch.object(torch, "compile", side_effect=AssertionError("compiler called")):
                with runtime.training_kernels(config):
                    pass

    def test_block_modes_merge_precision_rng_and_no_cudagraph_options(self):
        from torch import _inductor

        with patch.object(kernels, "validate_pointwise_support"):
            for mode in ("default", "max-autotune-no-cudagraphs"):
                expected_mode = dict(_inductor.list_mode_options(mode=mode, dynamic=True))
                options = kernels.validate_block_support(mode)
                self.assertIs(options["emulate_precision_casts"], True)
                self.assertIs(options["fallback_random"], True)
                self.assertIs(options["triton.cudagraphs"], False)
                self.assertIs(options["autotune_cudagraph_benchmarking"], False)
                if mode == "default":
                    for name in ("max_autotune", "max_autotune_gemm", "max_autotune_pointwise",
                                 "coordinate_descent_tuning"):
                        self.assertIs(options[name], False)
                else:
                    self.assertIs(options["max_autotune"], True)
                    for name, value in expected_mode.items():
                        if name not in ("emulate_precision_casts", "fallback_random", "triton.cudagraphs",
                                        "autotune_cudagraph_benchmarking"):
                            self.assertEqual(options[name], value)
                self.assertEqual(_inductor.list_mode_options(mode=mode, dynamic=True), expected_mode)
            with patch.object(_inductor, "list_mode_options", return_value={"triton.cudagraphs": True,
                               "emulate_precision_casts": False, "fallback_random": False}):
                options = kernels.validate_block_support("max-autotune-no-cudagraphs")
                self.assertFalse(options["triton.cudagraphs"])
                self.assertTrue(options["emulate_precision_casts"])
                self.assertTrue(options["fallback_random"])
            with patch.object(_inductor, "list_mode_options", None):
                with self.assertRaisesRegex(RuntimeError, "requires torch._inductor.list_mode_options"):
                    kernels.validate_block_support()

    def test_missing_block_option_and_disabled_dynamo_are_errors(self):
        from torch import _inductor
        from torch._dynamo import config as dynamo_config

        config = {"training": {"compile_pointwise": True, "compile_blocks": True},
                  "adapter": {"algo": "lora"}}
        with patch.object(kernels, "validate_pointwise_support"):
            options = [name for name in _inductor.list_options() if name != "fallback_random"]
            with patch.object(_inductor, "list_options", return_value=options):
                with self.assertRaisesRegex(RuntimeError, "requires Inductor options.*fallback_random"):
                    runtime.resolve_compile_blocks(config)
            with dynamo_config.patch(disable=True):
                with self.assertRaisesRegex(RuntimeError, "requires enabled Dynamo"):
                    runtime.resolve_compile_blocks(config)

    def test_disabled_blocks_leave_models_and_compiler_untouched(self):
        config = {"training": {"attention_backend": "auto", "compile_pointwise": False,
                               "compile_blocks": False}, "adapter": {"algo": "lokr"}}
        # Opaque objects prove the disabled path does not inspect/modify models.
        with patch.object(torch, "compile", side_effect=AssertionError("compiler called")):
            with runtime.training_kernels(config, dit=object(), network=object()):
                pass

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


class CudaBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CudaPointwiseTests.setUpClass()
        kernels.validate_block_support()

    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[torch.cuda.current_device()]))
        torch.random.default_generator.manual_seed(191)
        torch.cuda.manual_seed(191)

    def make_bundle(self, dropout):
        from lycoris.wrapper import LycorisNetwork

        # Same isolated real-LyCORIS preset pattern as test_checkpoint_groups.
        with torch.random.fork_rng(devices=[]), patch.multiple(
            LycorisNetwork, ENABLE_CONV=False, TARGET_REPLACE_MODULE=["_Block"],
            TARGET_REPLACE_NAME=[], LORA_PREFIX="lora_unet", MODULE_ALGO_MAP={},
            NAME_ALGO_MAP={}, USE_FNMATCH=False, TARGET_EXCLUDE_NAME=[],
        ):
            torch.random.default_generator.manual_seed(173)
            model = moss.MOSSImageTransformer(dim=128, ffn_dim=256, num_heads=4, num_layers=4,
                                              text_dim=64, freq_dim=32, latent_channels=8)
            model.requires_grad_(False)
            network = LycorisNetwork(model, lora_dim=4, alpha=4, network_module="lora",
                                     dropout=dropout, bypass_mode=True)
            network.apply_to()
            with torch.no_grad():
                for parameter in network.parameters():
                    parameter.normal_(mean=0.0, std=0.02)
        model.to(device="cuda", dtype=torch.bfloat16).train()
        network.to(device="cuda", dtype=torch.float32).requires_grad_(True).train()
        model._configure_checkpointing(True, 1)
        return model, network

    def exercise(self, model, network, values, target, state):
        network.zero_grad(set_to_none=True)
        torch.set_rng_state(state[0])
        torch.cuda.set_rng_state(state[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**values).sample
        loss = torch.nn.functional.mse_loss(output.float(), target)
        forward_rng = (torch.get_rng_state(), torch.cuda.get_rng_state())
        loss.backward()
        self.assertTrue(torch.equal(torch.get_rng_state(), forward_rng[0]))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), forward_rng[1]))
        gradients = {}
        for name, parameter in network.named_parameters():
            self.assertTrue(parameter.requires_grad)
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
            gradients[name] = parameter.grad.detach().clone()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        return output.detach(), loss.detach(), gradients, forward_rng

    def compare(self, reference, actual):
        close = CudaPointwiseTests.assert_numerically_close
        close(self, actual[0], reference[0])
        close(self, actual[1], reference[1])
        self.assertEqual(reference[2].keys(), actual[2].keys())
        for name in reference[2]:
            close(self, actual[2][name], reference[2][name], gradient=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[3], actual[3])))

    def test_real_lora_masks_rng_preview_and_worker_recomputation(self):
        from torch._dynamo import config as dynamo_config

        config = {"training": {"attention_backend": "auto", "compile_pointwise": True,
                               "compile_blocks": True}, "adapter": {"algo": "lora"}}
        self.assertIs(runtime.resolve_compile_blocks(config), True)
        generator = torch.Generator(device="cuda").manual_seed(179)
        values = {
            "hidden_states": torch.randn(2, 8, 4, 7, generator=generator, device="cuda"),
            "timestep": torch.tensor([100., 500.], device="cuda"),
            "encoder_hidden_states": torch.randn(2, 11, 64, generator=generator, device="cuda",
                                                  dtype=torch.bfloat16),
            "encoder_attention_mask": torch.tensor([[1] * 11, [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]],
                                                     device="cuda", dtype=torch.bool),
        }
        target = torch.randn(2, 8, 4, 7, generator=generator, device="cuda")
        for dropout in (0.0, 0.1):
            with self.subTest(dropout=dropout):
                baseline, baseline_network = self.make_bundle(dropout)
                model, network = self.make_bundle(dropout)
                frozen = {name: p.detach().clone() for name, p in model.named_parameters()}
                identities = tuple((name, id(p)) for name, p in network.named_parameters())
                base_identities = tuple((name, id(p)) for name, p in model.named_parameters())
                classes = tuple((name, type(module)) for name, module in model.named_modules())
                keys = tuple(model.state_dict()), tuple(network.state_dict())
                original_forwards = tuple(block.forward for block in model.layers)
                saved_functions, saved_cc = originals(), os.environ.get("CC")
                settings = ("specialize_float", "suppress_errors", "fail_on_recompile_limit_hit",
                            "skip_nnmodule_hook_guards")
                saved_settings = tuple(getattr(dynamo_config, key) for key in settings)
                with runtime.training_kernels(config, dit=model, network=network), ThreadPoolExecutor(1) as worker:
                    for batch in (1, 2):
                        inputs = {name: tensor[:batch] for name, tensor in values.items()}
                        state = torch.get_rng_state(), torch.cuda.get_rng_state()
                        reference = self.exercise(baseline, baseline_network, inputs, target[:batch], state)

                        def run_on_worker():
                            before = tuple(getattr(dynamo_config, key) for key in settings)
                            result = self.exercise(model, network, inputs, target[:batch], state)
                            self.assertEqual(tuple(getattr(dynamo_config, key) for key in settings), before)
                            return result

                        actual = worker.submit(run_on_worker).result()
                        self.compare(reference, actual)
                        if dropout:
                            self.assertFalse(torch.equal(state[1], actual[3][1]), "dropout was not exercised")
                        for dit, adapter in ((baseline, baseline_network), (model, network)):
                            dit.eval()
                            adapter.eval()
                            adapter.set_multiplier(0.5)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            expected_preview = baseline(**inputs).sample
                            actual_preview = model(**inputs).sample
                        CudaPointwiseTests.assert_numerically_close(self, actual_preview, expected_preview)
                        for dit, adapter in ((baseline, baseline_network), (model, network)):
                            dit.train()
                            adapter.train()
                            adapter.set_multiplier(1.0)
                        self.compare(reference, worker.submit(run_on_worker).result())
                    # In-place restored adapter values must be read by cached graphs.
                    with torch.no_grad():
                        for left, right in zip(baseline_network.parameters(), network.parameters()):
                            left.add_(0.01)
                            right.add_(0.01)
                    reference = self.exercise(baseline, baseline_network, inputs, target, state)
                    self.compare(reference, worker.submit(run_on_worker).result())
                self.assertEqual(tuple(block.forward for block in model.layers), original_forwards)
                self.assertEqual(originals(), saved_functions)
                self.assertEqual(os.environ.get("CC"), saved_cc)
                self.assertEqual(tuple(getattr(dynamo_config, key) for key in settings), saved_settings)
                self.assertEqual(tuple((name, id(p)) for name, p in network.named_parameters()), identities)
                self.assertEqual(tuple((name, id(p)) for name, p in model.named_parameters()), base_identities)
                self.assertEqual(tuple((name, type(module)) for name, module in model.named_modules()), classes)
                self.assertEqual((tuple(model.state_dict()), tuple(network.state_dict())), keys)
                self.assertTrue(all(torch.equal(p, frozen[name]) for name, p in model.named_parameters()))

    def test_block_scope_restores_after_exception_and_rejects_unapplied_hooks(self):
        config = {"training": {"attention_backend": "auto", "compile_pointwise": True,
                               "compile_blocks": True}, "adapter": {"algo": "lora"}}
        model, network = self.make_bundle(0.0)
        original_forwards, functions = tuple(block.forward for block in model.layers), originals()
        cc = os.environ.get("CC")
        with self.assertRaisesRegex(ArithmeticError, "scope sentinel"):
            with runtime.training_kernels(config, dit=model, network=network):
                raise ArithmeticError("scope sentinel")
        self.assertEqual(tuple(block.forward for block in model.layers), original_forwards)
        self.assertEqual(originals(), functions)
        self.assertEqual(os.environ.get("CC"), cc)
        adapter = network.loras[0]
        target, applied = adapter.org_module[0], adapter.org_module[0].forward
        target.forward = adapter.org_forward
        try:
            with self.assertRaisesRegex(RuntimeError, "after LyCORIS apply_to"):
                with runtime.training_kernels(config, dit=model, network=network):
                    self.fail("unapplied adapter was compiled")
        finally:
            target.forward = applied
        self.assertEqual(tuple(block.forward for block in model.layers), original_forwards)
        self.assertEqual(originals(), functions)


if __name__ == "__main__":
    unittest.main()
