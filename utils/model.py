from pathlib import Path
import torch
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel, load_checkpoint as load_checkpoint_weights

def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DSEC_NAMES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus',
    4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
}

def build_model(ch: int = 3, weights_path: str | Path | None = None) -> DetectionModel:
    """
    Build the stock YOLO26n architecture with a DSEC class head and `ch` input channels.
    Pretrained weights are transferred wherever layer names and shapes match.

    The teacher's first layer (RGB + 2 event channels) does not match the pretrained RGB one, so
    its RGB filters are copied from it and its event filters start at zero: the teacher starts as
    the pretrained RGB detector and learns from there how much to use the events.
    """
    model = DetectionModel("yolo26n.yaml", ch=ch, nc=len(DSEC_NAMES), verbose=False)
    if weights_path:
        pretrained = load_checkpoint_weights(str(weights_path))[0]
        model.load(pretrained)
        if ch > 3:
            with torch.no_grad():
                first_layer = model.model[0].conv.weight
                first_layer[:, :3] = pretrained.model[0].conv.weight
                first_layer[:, 3:] = 0
    model.names = DSEC_NAMES
    return model

def extract_trainable_model(
    ch: int = 3,
    weights_path: str | Path = "yolo26n.pt",
    device: str | torch.device | None = None,
    epochs: int = 1,
) -> torch.nn.Module:
    """Build a DSEC YOLO26n model with pretrained weights, ready for the custom training loop.

    The loss reads its gains (box, cls, dfl) from `model.args`, which gets Ultralytics' default
    training config just as its own trainer sets it; `epochs` drives the end-to-end loss schedule.
    """
    model = build_model(ch, weights_path).train()
    if device:
        model = model.to(device)
    model.args = get_cfg(overrides={"epochs": epochs})
    return model

def set_loss_schedule(model: torch.nn.Module, completed_epochs: int) -> None:
    if not hasattr(model, "init_criterion"):
        return
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    if hasattr(model.criterion, "update"):
        model.criterion.updates = completed_epochs - 1
        model.criterion.update()

NOMINAL_BATCH_SIZE = 64

def accumulation_steps(batch_size: int, nominal_batch_size: int = NOMINAL_BATCH_SIZE) -> int:
    """Batches per optimiser step once warmup is over."""
    return max(round(nominal_batch_size / batch_size), 1)

OPTIMISER_LR = 0.01
OPTIMISER_MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

def create_optimiser(
    model: torch.nn.Module, batch_size: int = NOMINAL_BATCH_SIZE
) -> torch.optim.Optimizer:
    """MuSGD with Ultralytics' own YOLO26 parameter groups, built by its trainer's optimiser factory.

    Conv/linear weights get the hybrid Muon + SGD update with weight decay, normalisation weights
    and biases plain SGD without it, and the head's classification layers 3x the learning rate.
    Weight decay is scaled by the images per optimiser step, as the Ultralytics trainer does.
    """
    decay = WEIGHT_DECAY * batch_size * accumulation_steps(batch_size) / NOMINAL_BATCH_SIZE
    # build_optimizer only reads `self` when picking an optimiser for name="auto", so no trainer is needed
    return BaseTrainer.build_optimizer(
        None, model, name="MuSGD", lr=OPTIMISER_LR, momentum=OPTIMISER_MOMENTUM, decay=decay
    )


def load_weights(
    model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device | str
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    return model


def load_checkpoint(
    checkpoint_path: str | Path, 
    device: torch.device, 
    model: torch.nn.Module, 
    optimiser: torch.optim.Optimizer,
    ema,
    scheduler,
) -> int:
    """Restore the latest checkpoint if there is one, and return how many epochs it completed.

    The checkpoint holds the raw model, the optimiser, the EMA model and its update count, and the
    learning-rate schedule, so training continues exactly where it stopped.
    """
    if not Path(checkpoint_path).is_file():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["raw_model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    ema.ema.load_state_dict(checkpoint["model_state_dict"])
    ema.updates = checkpoint["ema_updates"]
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]

def best_checkpoint_map50_95(best_path: str | Path, device: torch.device) -> float:
    """Validation mAP50-95 of the best checkpoint so far, or -inf before there is one."""
    if not Path(best_path).is_file():
        return float('-inf')
    return torch.load(best_path, map_location=device)["val_metrics"]["map50_95"]