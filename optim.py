"""Native PyTorch optimizers and diffusers learning-rate schedules."""

from __future__ import annotations

import torch
from diffusers.optimization import get_scheduler


def make_optimizers(network, config: dict) -> dict[str, torch.optim.Optimizer]:
    """Use Muon only for matrices; other trainable tensors use native AdamW."""
    settings = config["optimizer"]
    parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
    learning_rate = float(settings["learning_rate"])
    weight_decay = float(settings["weight_decay"])
    kwargs = settings["kwargs"]
    if settings["type"] == "adamw":
        return {
            "adamw": torch.optim.AdamW(
                parameters, lr=learning_rate, weight_decay=weight_decay, **kwargs
            )
        }
    if settings["type"] == "muon":
        matrices = [parameter for parameter in parameters if parameter.ndim == 2]
        auxiliary = [parameter for parameter in parameters if parameter.ndim != 2]
        optimizers = {}
        if matrices:
            optimizers["muon"] = torch.optim.Muon(
                matrices, lr=learning_rate, weight_decay=weight_decay, **kwargs
            )
        if auxiliary:
            optimizers["adamw_aux"] = torch.optim.AdamW(
                auxiliary,
                lr=float(settings.get("aux_learning_rate", learning_rate)),
                weight_decay=weight_decay,
            )
        return optimizers
    raise ValueError(f"unsupported optimizer type: {settings['type']!r}")


def make_schedulers(optimizers: dict, config: dict, total_steps: int) -> dict:
    """A constant schedule with warmup uses diffusers' constant_with_warmup."""
    settings = config["scheduler"]
    name = settings["type"]
    warmup_steps = int(settings["warmup_steps"])
    if name == "constant" and warmup_steps:
        name = "constant_with_warmup"
    return {
        key: get_scheduler(
            name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        for key, optimizer in optimizers.items()
    }
