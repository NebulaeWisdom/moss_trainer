"""Real tiny MOSS/LoRA contracts; CPU FP32 needs no optional compiler."""

from functools import partial
import json
import unittest
from unittest.mock import patch

from moss_trainer.launch import configure_environment

configure_environment()  # Before model imports: runtime caches stay in temp/.

import torch
from lycoris.wrapper import LycorisNetwork
from torch.utils.checkpoint import checkpoint

from moss_trainer import runtime
from moss_trainer.vendor.pipeline_moss_image import MOSSImageTransformer


def make_bundle(device, *, depth=4, dropout=0.1, legacy=False):
    # Initialize on CPU with a local RNG scope, including nonzero LoRA up weights.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(173)
        model = MOSSImageTransformer(dim=16, ffn_dim=24, num_heads=2, num_layers=depth,
                                     text_dim=12, freq_dim=8, latent_channels=4)
        model.requires_grad_(False)
        base = dict(model.named_parameters())
        keys = tuple(model.state_dict())
        classes = tuple((name, type(module)) for name, module in model.named_modules())
        # Match the production _Block-only preset without leaking LyCORIS class settings.
        with patch.multiple(LycorisNetwork, ENABLE_CONV=False, TARGET_REPLACE_MODULE=["_Block"],
                            TARGET_REPLACE_NAME=[], LORA_PREFIX="lora_unet", MODULE_ALGO_MAP={},
                            NAME_ALGO_MAP={}, USE_FNMATCH=False, TARGET_EXCLUDE_NAME=[]):
            network = LycorisNetwork(model, lora_dim=2, alpha=2, network_module="lora",
                                    dropout=dropout, bypass_mode=True)
        network.apply_to()
        network.requires_grad_(True)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.normal_(mean=0.0, std=0.05)
    model.to(device).train()
    network.to(device).train()
    owners = {}
    layer_modules = [{id(module) for module in layer.modules()} for layer in model.layers]
    for adapter in network.loras:
        indices = [index for index, modules in enumerate(layer_modules)
                   if id(adapter.org_module[0]) in modules]
        if len(indices) != 1:
            raise AssertionError("Each adapter must target exactly one main block")
        owners.update((id(parameter), indices[0]) for parameter in adapter.parameters())
    if legacy:
        # The original runtime's per-block wrappers are an independent reference.
        for block in (*model.layers, *model.text_embedder.blocks):
            block.forward = partial(checkpoint, block.forward, use_reentrant=False,
                                    preserve_rng_state=True)
    return {"model": model, "network": network, "base": base, "keys": keys, "classes": classes,
            "frozen": {name: parameter.detach().clone() for name, parameter in base.items()},
            "owners": owners}


def inputs(device):
    generator = torch.Generator().manual_seed(179)
    return {
        "hidden_states": torch.randn(2, 4, 2, 2, generator=generator).to(device),
        "timestep": torch.tensor([100.0, 700.0], device=device),
        "encoder_hidden_states": torch.randn(2, 5, 12, generator=generator).to(device),
        "encoder_attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 0, 1, 1]],
                                                dtype=torch.bool, device=device),
    }


def rng_state(device):
    states = [torch.get_rng_state().clone()]
    if device == "cuda":
        states.append(torch.cuda.get_rng_state().clone())
    return states


def restore_rng(device, states):
    torch.set_rng_state(states[0])
    if device == "cuda":
        torch.cuda.set_rng_state(states[1])


class ConfigurationTests(unittest.TestCase):
    def test_missing_stored_setting_defaults_to_one_and_roundtrips(self):
        for enabled in (False, True):
            config = json.loads(json.dumps({"training": {"gradient_checkpointing": enabled}}))
            self.assertEqual(runtime.resolve_checkpoint_group_size(config), 1)
            self.assertEqual(json.loads(json.dumps(config))["training"]["checkpoint_group_size"], 1)
            for size in (1, 2, 3, 43):
                config["training"]["checkpoint_group_size"] = size
                self.assertEqual(runtime.resolve_checkpoint_group_size(config), size)
                self.assertEqual(json.loads(json.dumps(config))["training"]["checkpoint_group_size"], size)

    def test_invalid_groups_fail_before_model_loading_even_when_disabled(self):
        for invalid in (True, False, 0, -1, 1.0, 2.5, "2", None, [], {}):
            with self.subTest(invalid=invalid):
                config = {"training": {"checkpoint_group_size": invalid}}
                with self.assertRaisesRegex(ValueError, "training.checkpoint_group_size"):
                    runtime.resolve_checkpoint_group_size(config)
                # No model_dir is supplied: validation must precede any model access.
                with self.assertRaisesRegex(ValueError, "training.checkpoint_group_size"):
                    runtime.load_dit(config, checkpointing=False)


class CheckpointAssertions:
    device = "cpu"

    def setUp(self):
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        self.enterContext(torch.random.fork_rng(devices=devices))
        torch.random.default_generator.manual_seed(191)
        if self.device == "cuda":
            torch.cuda.manual_seed(191)

    def assert_rng_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            self.assertTrue(torch.equal(left, right), "checkpoint changed caller RNG")

    def exercise(self, bundle, values, start_rng, skip=(), reconfigure=False):
        model, network = bundle["model"], bundle["network"]
        parameters = dict(network.named_parameters())
        self.assertTrue(parameters)
        self.assertEqual({id(parameter) for parameter in parameters.values()}, set(bundle["owners"]))
        network.zero_grad(set_to_none=True)
        restore_rng(self.device, start_rng)
        output = model(**values, skip_layers=skip).sample
        forward_rng = rng_state(self.device)
        if reconfigure:
            # Pending backward must retain its original immutable group indices.
            model._configure_checkpointing(True, 3)
        output.square().mean().backward()
        self.assert_rng_equal(rng_state(self.device), forward_rng)
        gradients = {}
        for name, parameter in parameters.items():
            if bundle["owners"][id(parameter)] in skip:
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)
                gradients[name] = parameter.grad.detach().clone()
        for name, parameter in bundle["base"].items():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(parameter, bundle["frozen"][name]), name)
        self.assertEqual(tuple(model.state_dict()), bundle["keys"])
        self.assertEqual(tuple((name, type(module)) for name, module in model.named_modules()),
                         bundle["classes"])
        return output.detach(), gradients, forward_rng

    def assert_result_equal(self, actual, expected):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertEqual(actual[1].keys(), expected[1].keys())
        for name in expected[1]:
            torch.testing.assert_close(actual[1][name], expected[1][name], rtol=1e-6, atol=1e-7,
                                       msg=lambda message, name=name: f"{name}: {message}")
        self.assert_rng_equal(actual[2], expected[2])

    def compare_groups(self, *, depth=4, skip=(), dropout=0.1, sizes=(1, 2, 3)):
        values = inputs(self.device)
        # The frozen first block input has no gradient; adapters still must receive it.
        self.assertTrue(all(not value.requires_grad for value in values.values()))
        reference = make_bundle(self.device, depth=depth, dropout=dropout, legacy=True)
        start_rng = rng_state(self.device)
        expected = self.exercise(reference, values, start_rng, skip)
        if dropout:
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(start_rng, expected[2])),
                            "nonzero adapter dropout did not consume RNG")
        for size in sizes:
            with self.subTest(group_size=size, depth=depth, skip=skip, dropout=dropout):
                candidate = make_bundle(self.device, depth=depth, dropout=dropout)
                candidate["model"]._configure_checkpointing(True, size)
                self.assertEqual(tuple(candidate["network"].state_dict()),
                                 tuple(reference["network"].state_dict()))
                actual = self.exercise(candidate, values, start_rng, skip, reconfigure=size == 2)
                self.assert_result_equal(actual, expected)


class CpuCheckpointTests(CheckpointAssertions, unittest.TestCase):
    def test_groups_match_legacy_for_all_adapter_gradients_and_dropout(self):
        for dropout in (0.0, 0.1):
            self.compare_groups(dropout=dropout)

    def test_skip_one_omits_only_its_adapter_gradients(self):
        self.compare_groups(skip=(1,))

    def test_skip_twenty_two_and_one_use_active_index_order(self):
        self.compare_groups(depth=23, skip=(1, 22), sizes=(3,))

    def test_disabled_checkpointing_ignores_group_size(self):
        values = inputs(self.device)
        start_rng = rng_state(self.device)
        model = make_bundle(self.device)
        expected = self.exercise(model, values, start_rng)
        for size in (1, 2, 3, 7):
            with self.subTest(group_size=size):
                model["model"]._configure_checkpointing(False, size)
                self.assert_result_equal(self.exercise(model, values, start_rng), expected)

    def test_no_grad_and_inference_preserve_outputs_and_masked_tokens(self):
        bundle = make_bundle(self.device)
        model, network = bundle["model"].eval(), bundle["network"].eval()
        values = inputs(self.device)
        poisoned = dict(values)
        poisoned["encoder_hidden_states"] = values["encoder_hidden_states"].clone()
        poisoned["encoder_hidden_states"][~values["encoder_attention_mask"]] = 1e6
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                for skip in ((), (1,), (0, 1, 2, 3)):
                    model._configure_checkpointing(False, 1)
                    expected = model(**values, skip_layers=skip).sample
                    for size in (1, 2, 3):
                        with self.subTest(context=context.__name__, skip=skip, group_size=size):
                            model._configure_checkpointing(True, size)
                            actual = model(**poisoned, skip_layers=skip).sample
                            self.assertFalse(actual.requires_grad)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in network.parameters()))


class CudaCheckpointTests(CheckpointAssertions, unittest.TestCase):
    device = "cuda"

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("optional real-CUDA grouped checkpoint contract")

    def test_real_cuda_groups_and_dropout(self):
        self.compare_groups(skip=(1,))


if __name__ == "__main__":
    unittest.main()
