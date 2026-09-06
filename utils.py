from pathlib import Path
from types import SimpleNamespace
import torch
from ultralytics import YOLO
import yaml

def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_trainable_model(
    yaml_path: str | Path,
    weights_path: str | Path = "yolo26n.pt", 
    device: str | torch.device | None = None
) -> torch.nn.Module:
    """
    Load a YOLO architecture from a YAML file, load pretrained weights,
    extract the raw PyTorch nn.Module, and prep it for custom training.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ch = cfg.get('ch', 3)

    yolo_wrapper = YOLO(str(yaml_path), task="detect")
    
    if ch != 3:
        from ultralytics.nn.tasks import DetectionModel
        yolo_wrapper.model = DetectionModel(str(yaml_path), ch=ch, nc=cfg.get('nc', 80))
        
    yolo_wrapper.load(str(weights_path))
    
    dsec_names = {
        0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'bus', 
        4: 'truck', 5: 'bicycle', 6: 'motorcycle', 7: 'train'
    }
    yolo_wrapper.model.names = dsec_names
    
    model = yolo_wrapper.model
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
        'device': str(device) if device else None
    }
    
    if hasattr(model, 'args') and isinstance(model.args, dict):
        training_hyps.update(model.args)

    hyp_namespace = SimpleNamespace(**training_hyps)
    
    model.args = hyp_namespace
    for m in model.modules():
        if hasattr(m, 'args'):
            m.args = hyp_namespace

    return model

def create_optimiser(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create the optimiser for the unfrozen student parameters."""
    #return torch.optim.SGD(
    #    (parameter for parameter in model.parameters() if parameter.requires_grad), 
    #    lr=5e-2, 
    #    momentum=0.9, 
    #    weight_decay=1e-4
    #)
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=3e-4,
        weight_decay=0.0005,
    )
    

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
    """ Load the best checkpoint if it exists, otherwise return default values. """
    if not Path(best_path).is_file():
        return {'best_epoch': 0, 'best_val_loss': float('inf')}

    best_state = torch.load(best_path, map_location=device)

    if mode == 1:
        model.load_state_dict(best_state["model_state_dict"])
        optimiser.load_state_dict(best_state["optimiser_state_dict"])

    return {'best_epoch': best_state["epoch"], 'best_val_loss': best_state["val_loss"]}