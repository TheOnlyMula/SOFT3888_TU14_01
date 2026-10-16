"""Learning-rate, momentum and gradient-accumulation schedule of the training loop.

Warmup and schedule follow Ultralytics' YOLO26 defaults ("optimizer=auto" picking MuSGD).
Over the warmup the learning rate rises from 0, momentum rises from WARMUP_MOMENTUM, and the
images per optimiser step rise from one batch to NOMINAL_BATCH_SIZE.
"""
import numpy as np
import torch

WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
MOMENTUM_AFTER_WARMUP = 0.937
FINAL_LR_FRACTION = 0.01


def lr_factor(epoch: int, total_epochs: int) -> float:
    """Fraction of the starting learning rate used in `epoch` (Ultralytics' linear schedule)."""
    return max(1 - epoch / total_epochs, 0) * (1 - FINAL_LR_FRACTION) + FINAL_LR_FRACTION


def apply_warmup(
    optimiser: torch.optim.Optimizer,
    batch_index: int,
    warmup_batches: int,
    epoch_lr_factor: float,
    nominal_accumulate: float,
) -> int:
    """Set one warmup batch's learning rate and momentum, and return its accumulation steps.

    All three ramp linearly over the first `warmup_batches` batches: the learning rate from 0 to
    this epoch's scheduled rate, momentum from WARMUP_MOMENTUM to MOMENTUM_AFTER_WARMUP, and the
    batches per optimiser step from 1 to `nominal_accumulate`.
    """
    ramp = [0, warmup_batches]
    accumulate = max(1, round(float(np.interp(batch_index, ramp, [1, nominal_accumulate]))))
    for group in optimiser.param_groups:
        target_lr = group["initial_lr"] * epoch_lr_factor
        group["lr"] = float(np.interp(batch_index, ramp, [0.0, target_lr]))
        group["momentum"] = float(
            np.interp(batch_index, ramp, [WARMUP_MOMENTUM, MOMENTUM_AFTER_WARMUP])
        )
    return accumulate
