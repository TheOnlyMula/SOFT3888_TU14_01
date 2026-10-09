import os
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel, load_checkpoint as load_checkpoint_weights
from ultralytics.optim import MuSGD

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

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch so model init, shuffling and augmentation repeat.

    With `deterministic`, also request deterministic GPU kernels the way Ultralytics' `init_seeds`
    does. Ops without a deterministic version only warn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

NOMINAL_BATCH_SIZE = 64

def accumulation_steps(batch_size: int, nominal_batch_size: int = NOMINAL_BATCH_SIZE) -> int:
    """Batches per optimiser step once warmup is over."""
    return max(round(nominal_batch_size / batch_size), 1)

OPTIMISER_LR = 0.01
OPTIMISER_MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MUON_WEIGHT, SGD_WEIGHT = 0.2, 1.0
CLASSIFICATION_HEAD_LR_GAIN = 3.0

def create_optimiser(
    model: torch.nn.Module, batch_size: int = NOMINAL_BATCH_SIZE
) -> torch.optim.Optimizer:
    """Create MuSGD for the unfrozen parameters with Ultralytics' YOLO26 parameter groups."""
    norm_layers = tuple(layer for name, layer in torch.nn.__dict__.items() if "Norm" in name)
    weights, norm_weights, biases, muon_weights = [], [], [], []
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if parameter.ndim in {2, 4}:  
                muon_weights.append(parameter)
            elif name == "bias":
                biases.append(parameter)
            elif isinstance(module, norm_layers):
                norm_weights.append(parameter)
            else:
                weights.append(parameter)

    decay = WEIGHT_DECAY * batch_size * accumulation_steps(batch_size) / NOMINAL_BATCH_SIZE
    shared = {"lr": OPTIMISER_LR, "momentum": OPTIMISER_MOMENTUM, "nesterov": True}
    groups = [
        {"params": weights, **shared, "weight_decay": decay, "param_group": "weight"},
        {"params": norm_weights, **shared, "weight_decay": 0.0, "param_group": "bn"},
        {"params": biases, **shared, "weight_decay": 0.0, "param_group": "bias"},
        {"params": muon_weights, **shared, "weight_decay": decay, "use_muon": True, "param_group": "muon"},
    ]

    head = getattr(model, "model", [None])[-1]
    boosted = {
        id(parameter)
        for layer in (getattr(head, "cv3", None), getattr(head, "one2one_cv3", None))
        if layer is not None
        for parameter in layer.parameters()
    }
    split_groups = []
    for group in groups:
        params = group.pop("params")
        split_groups.append(
            {
                **group,
                "params": [p for p in params if id(p) in boosted],
                "lr": OPTIMISER_LR * CLASSIFICATION_HEAD_LR_GAIN,
            }
        )
        split_groups.append({**group, "params": [p for p in params if id(p) not in boosted]})

    return MuSGD(split_groups, muon=MUON_WEIGHT, sgd=SGD_WEIGHT)


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