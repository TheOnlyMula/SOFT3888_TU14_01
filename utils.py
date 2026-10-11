from pathlib import Path
from types import SimpleNamespace
import torch
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
    """
    model = DetectionModel("yolo26n.yaml", ch=ch, nc=len(DSEC_NAMES), verbose=False)
    if weights_path:
        model.load(load_checkpoint_weights(str(weights_path))[0])
    model.names = DSEC_NAMES
    return model

def extract_trainable_model(
    ch: int = 3,
    weights_path: str | Path = "yolo26n.pt", 
    device: str | torch.device | None = None,
    epochs: int = 1,
) -> torch.nn.Module:
    """
    Build a DSEC YOLO26n model, load pretrained weights,
    and prep the raw PyTorch nn.Module for custom training.
    """
    model = build_model(ch, weights_path)
    model.train()
    
    if device:
        model = model.to(device)
        
    for param in model.parameters():
        param.requires_grad = True

    training_hyps = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.0,
        'fl_gamma': 2.0,
        'device': str(device) if device else None,
        'epochs': epochs,
    }

    hyp_namespace = SimpleNamespace(**training_hyps)
    
    model.args = hyp_namespace
    for m in model.modules():
        if hasattr(m, 'args'):
            m.args = hyp_namespace

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
    ema=None,
    scheduler=None
) -> dict:
    """ Load the checkpoint if it exists, otherwise return default values. """
    if not Path(checkpoint_path).is_file():
        return {'past_epochs': 0, 'latest_val_loss': float('inf')}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "raw_model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["raw_model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
    
    if ema is not None and "ema_updates" in checkpoint:
        ema.ema.load_state_dict(checkpoint["model_state_dict"])
        ema.updates = checkpoint["ema_updates"]
        
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {'past_epochs': checkpoint["epoch"], 'latest_val_loss': checkpoint["val_loss"]}

def fetch_best_dict(
    best_path: str | Path, 
    device: torch.device, 
    model: torch.nn.Module, 
    optimiser: torch.optim.Optimizer, 
    mode: int = 0
) -> dict:
    """ Load the best checkpoint if it exists, otherwise return default values."""
    if not Path(best_path).is_file():
        return {'best_epoch': 0, 'best_map50_95': float('-inf')}

    best_state = torch.load(best_path, map_location=device)

    if mode == 1:
        model.load_state_dict(best_state["model_state_dict"])
        optimiser.load_state_dict(best_state["optimiser_state_dict"])

    best_map50_95 = best_state.get("val_metrics", {}).get("map50_95", float('-inf'))
    return {'best_epoch': best_state["epoch"], 'best_map50_95': best_map50_95}