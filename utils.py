from pathlib import Path
from types import SimpleNamespace
import torch
from ultralytics import YOLO

def select_device() -> torch.device:
    """Select CUDA when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_trainable_model(weights_path: str = "yolo26n.pt", device: str | None = None) -> torch.nn.Module:
    """
    Load a pretrained YOLO wrapper, extract the raw PyTorch nn.Module,
    prep it for custom training, and bind loss hyperparameters.

    Returns:
        model (torch.nn.Module): the raw detection model, ready to be
        used directly in a custom PyTorch training loop.
    """
    yolo_wrapper = YOLO("yolo26n.yaml", task="detect")
    yolo_wrapper.load(weights_path)
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

    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        fl_gamma=2,
        device=device
    )

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
        lr=1e-4,
        weight_decay=0.0005,
    )
    

def load_checkpoint(
    checkpoint_path: str | Path, 
    device: torch.device, 
    model: torch.nn.Module, 
    optimiser: torch.optim.Optimizer
) -> dict:
    """ Load the checkpoint if it exists, otherwise return default values. """
    if not Path(checkpoint_path).is_file():
        return {'past_epochs': 0, 'latest_val_loss': float('inf')}

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

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